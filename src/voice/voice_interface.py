# src/voice/voice_interface.py
import os
import json
import numpy as np
import sounddevice as sd
import queue
import threading
import time
from vosk import Model, KaldiRecognizer
import torch
import io
import wave
import tempfile
import struct
import scipy.signal  # для ресемплинга

class VoiceInterface:
    def __init__(self, model_name="vosk-model-small-ru-0.22"):
        """
        Инициализация голосового интерфейса.
        
        Args:
            model_name: название модели Vosk (должна лежать в models/vosk/)
        """
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.target_sample_rate = 16000  # То, что ожидает Vosk
        
        # Путь к модели Vosk
        self.model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "models",
            "vosk",
            model_name
        )
        
        # Инициализация Vosk (распознавание речи)
        print("Загрузка модели Vosk...")
        if os.path.exists(self.model_path):
            self.stt_model = Model(self.model_path)
            print(f"✓ Модель Vosk загружена из {self.model_path}")
        else:
            print(f"⚠ Модель Vosk не найдена в {self.model_path}")
            print("  Скачайте модель с https://alphacephei.com/vosk/models")
            print("  и поместите в models/vosk/")
            self.stt_model = None
        
        # Инициализация Silero (синтез речи)
        print("Загрузка модели Silero TTS...")
        try:
            # Указываем путь для кэша в папку models
            torch.hub.set_dir(os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "models",
                "silero"
            ))
            
            self.tts_model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language='ru',
                speaker='v4_ru'
            )
            self.tts_model.to('cpu')
            print("✓ Модель Silero загружена")
        except Exception as e:
            print(f"⚠ Ошибка загрузки Silero: {e}")
            self.tts_model = None
        
        self.speakers = ['baya', 'kseniya', 'xenia', 'random']
        self.current_speaker = 'baya'
        
        # Для отладки сохраним несколько записей
        self.debug_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "debug_audio")
        os.makedirs(self.debug_dir, exist_ok=True)
        self.debug_counter = 0
    
    def resample_audio(self, audio_data, original_sr, target_sr=16000):
        """Передискретизация аудио с оригинальной частоты на целевую"""
        if original_sr == target_sr:
            return audio_data
        
        print(f"🔄 Ресемплинг с {original_sr}Hz на {target_sr}Hz")
        
        # Определяем тип данных
        dtype = audio_data.dtype
        
        # Конвертируем в float64 для обработки
        if dtype == np.int16:
            audio_float = audio_data.astype(np.float64) / 32768.0
        else:
            audio_float = audio_data.astype(np.float64)
        
        # Вычисляем новую длину
        new_length = int(len(audio_float) * target_sr / original_sr)
        
        # Ресемплинг
        resampled = scipy.signal.resample(audio_float, new_length)
        
        # Конвертируем обратно в int16
        resampled_int16 = (resampled * 32767).astype(np.int16)
        
        print(f"   Было: {len(audio_data)} сэмплов, {original_sr}Hz")
        print(f"   Стало: {len(resampled_int16)} сэмплов, {target_sr}Hz")
        
        return resampled_int16
    
    def record_audio(self, duration=5):
        """Запись аудио с микрофона (для теста) - сразу на 16kHz"""
        print(f"🎤 Запись... (говорите {duration} сек)")
        recording = sd.rec(
            int(duration * self.target_sample_rate),
            samplerate=self.target_sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        print("✓ Запись завершена")
        return recording.flatten()
    
    def debug_save_audio(self, audio_data, sample_rate, prefix="debug"):
        """Сохраняет аудио для отладки"""
        try:
            self.debug_counter += 1
            filename = os.path.join(self.debug_dir, f"{prefix}_{self.debug_counter}.wav")
            
            # Конвертируем в int16 для сохранения
            if isinstance(audio_data, np.ndarray):
                if audio_data.dtype == np.float32:
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                else:
                    audio_int16 = audio_data.astype(np.int16)
                
                # Сохраняем как WAV
                import scipy.io.wavfile as wav
                wav.write(filename, sample_rate, audio_int16)
                print(f"💾 Аудио сохранено: {filename} ({sample_rate}Hz)")
        except Exception as e:
            print(f"⚠ Не удалось сохранить аудио: {e}")
    
    def inspect_audio(self, audio_data, context=""):
        """Детальный анализ аудиоданных"""
        print(f"\n=== Анализ аудио {context} ===")
        print(f"Тип: {type(audio_data)}")
        
        if isinstance(audio_data, tuple) and len(audio_data) == 2:
            sr, data = audio_data
            print(f"Формат: кортеж (sample_rate={sr}, data)")
            audio_data = data
        
        if isinstance(audio_data, np.ndarray):
            print(f"Форма: {audio_data.shape}")
            print(f"dtype: {audio_data.dtype}")
            print(f"Размер в байтах: {audio_data.nbytes}")
            print(f"min: {audio_data.min():.6f}")
            print(f"max: {audio_data.max():.6f}")
            print(f"mean: {audio_data.mean():.6f}")
            print(f"std: {audio_data.std():.6f}")
            print(f"Количество нулевых значений: {np.sum(audio_data == 0)}")
            print(f"Первые 10 значений: {audio_data.flatten()[:10]}")
        elif isinstance(audio_data, bytes):
            print(f"Длина в байтах: {len(audio_data)}")
            print(f"Первые 20 байт: {audio_data[:20]}")
        else:
            print(f"Содержимое: {audio_data}")
        print("=" * 30)
    
    def extract_audio_data_and_rate(self, audio_input):
        """Извлекает аудиоданные и частоту дискретизации из разных форматов"""
        sample_rate = self.target_sample_rate  # По умолчанию
        audio_data = None
        
        if isinstance(audio_input, tuple) and len(audio_input) == 2:
            # Формат Gradio: (sample_rate, data)
            sample_rate, audio_data = audio_input
            print(f"📦 Формат Gradio: {sample_rate}Hz, данные {type(audio_data)}")
        
        elif isinstance(audio_input, dict):
            # Альтернативный формат
            sample_rate = audio_input.get('sample_rate', self.target_sample_rate)
            audio_data = audio_input.get('data', audio_input.get('array'))
            print(f"📦 Формат dict: {sample_rate}Hz")
        
        elif isinstance(audio_input, np.ndarray):
            # Только данные, частота неизвестна
            audio_data = audio_input
            print(f"📦 Только numpy массив, частота неизвестна, используем {sample_rate}Hz")
        
        return sample_rate, audio_data
    
    def transcribe(self, audio_input):
        """
        Распознавание речи с правильной обработкой частоты дискретизации.
        """
        if self.stt_model is None:
            print("❌ Модель Vosk не загружена")
            return None
        
        print("\n" + "="*50)
        print("🔄 НАЧАЛО РАСПОЗНАВАНИЯ")
        print("="*50)
        
        # 1. Извлекаем данные и частоту
        original_rate, audio_data = self.extract_audio_data_and_rate(audio_input)
        
        if audio_data is None:
            print("❌ Не удалось извлечь аудиоданные")
            return None
        
        # 2. Инспектируем исходные данные
        self.inspect_audio(audio_data, f"ИСХОДНЫЕ (частота {original_rate}Hz)")
        
        # 3. Сохраняем оригинал для отладки
        self.debug_save_audio(audio_data, original_rate, "original")
        
        # 4. Конвертируем в int16 если нужно
        if audio_data.dtype != np.int16:
            if audio_data.dtype == np.float32:
                # Нормализуем если нужно
                if audio_data.max() <= 1.0 and audio_data.min() >= -1.0:
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                else:
                    audio_int16 = audio_data.astype(np.int16)
            else:
                audio_int16 = audio_data.astype(np.int16)
        else:
            audio_int16 = audio_data
        
        # 5. Ресемплинг если нужно (самое важное!)
        if original_rate != self.target_sample_rate:
            audio_int16 = self.resample_audio(audio_int16, original_rate, self.target_sample_rate)
        
        # 6. Убеждаемся что это 1D массив
        if len(audio_int16.shape) > 1:
            audio_int16 = audio_int16.flatten()
        
        # 7. Проверяем длительность
        duration = len(audio_int16) / self.target_sample_rate
        print(f"Длительность: {duration:.2f} сек")
        
        if duration < 0.3:  # Уменьшил порог до 0.3 сек
            print("⚠ Слишком короткая запись")
            return None
        
        # 8. Конвертируем в байты
        audio_bytes = audio_int16.tobytes()
        
        # 9. Сохраняем финальную версию
        self.debug_save_audio(audio_int16, self.target_sample_rate, "resampled")
        
        # 10. Запускаем Vosk
        print("\n🔍 Запуск Vosk...")
        
        # Пробуем с новым recognizer
        rec = KaldiRecognizer(self.stt_model, self.target_sample_rate)
        rec.SetWords(True)
        
        if rec.AcceptWaveform(audio_bytes):
            result = json.loads(rec.Result())
            text = result.get('text', '')
            print(f"\n✅ РЕЗУЛЬТАТ: '{text}'")
            print("="*50)
            return text if text else None
        
        # Если не получилось, пробуем через файл
        print("\n🔄 Пробуем распознавание через файл...")
        try:
            wav_filename = os.path.join(self.debug_dir, f"vosk_input_{self.debug_counter}.wav")
            with wave.open(wav_filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.target_sample_rate)
                wf.writeframes(audio_bytes)
            
            wf = wave.open(wav_filename, 'rb')
            rec2 = KaldiRecognizer(self.stt_model, wf.getframerate())
            
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                rec2.AcceptWaveform(data)
            
            result = json.loads(rec2.FinalResult())
            text = result.get('text', '')
            print(f"📝 Через файл: '{text}'")
            print("="*50)
            return text if text else None
            
        except Exception as e:
            print(f"⚠ Ошибка при распознавании через файл: {e}")
            print("="*50)
            return None
    
    # def synthesize(self, text, speaker=None, max_length=1000):
    #     """Синтез речи из текста (без изменений)"""
    #     if self.tts_model is None:
    #         return None
        
    #     if speaker is None:
    #         speaker = self.current_speaker
        
    #     if len(text) > max_length:
    #         text = text[:max_length] + "..."
    #         print(f"⚠ Текст обрезан до {max_length} символов")
        
    #     print(f"🔊 Синтез речи ({len(text)} символов)...")
    #     try:
    #         audio = self.tts_model.apply_tts(
    #             text=text,
    #             speaker=speaker,
    #             sample_rate=24000
    #         )
            
    #         audio_numpy = audio.numpy().astype(np.float32)
            
    #         if np.abs(audio_numpy).max() > 1.0:
    #             audio_numpy = audio_numpy / 32768.0
            
    #         print(f"✓ Готово: {len(audio_numpy)} сэмплов")
    #         return (24000, audio_numpy)
            
    #     except Exception as e:
    #         print(f"⚠ Ошибка синтеза: {e}")
    #         return None]

    def synthesize(self, text, speaker=None, max_length=500):
        """
        Синтез речи из текста с ограничением длины.
        
        Args:
            text: текст для озвучивания
            speaker: голос (baya, kseniya, xenia)
            max_length: максимальная длина текста
            
        Returns:
            tuple (sample_rate, audio_data) или None
        """
        if self.tts_model is None:
            print("⚠ TTS модель не загружена")
            return None
        
        if speaker is None:
            speaker = self.current_speaker
        
        # Проверяем что текст не пустой
        if not text or len(text.strip()) == 0:
            print("⚠ Пустой текст для синтеза")
            return None
        
        # Очищаем текст от лишних символов
        text = text.strip()
        
        # Ограничиваем длину текста
        if len(text) > max_length:
            # Ищем конец предложения для более естественного обрезания
            truncated = text[:max_length]
            last_period = truncated.rfind('.')
            last_question = truncated.rfind('?')
            last_exclamation = truncated.rfind('!')
            
            cut_point = max(last_period, last_question, last_exclamation)
            if cut_point > max_length * 0.7:  # Если нашли знак препинания в последней трети
                text = text[:cut_point + 1]
            else:
                text = truncated + '...'
            
            print(f"⚠ Текст обрезан до {len(text)} символов")
        
        print(f"🔊 Синтез речи ({len(text)} символов)...")
        try:
            # Пробуем синтезировать
            audio = self.tts_model.apply_tts(
                text=text,
                speaker=speaker,
                sample_rate=24000
            )
            
            # Проверяем что аудио не пустое
            if audio is None or len(audio) == 0:
                print("⚠ Синтезированное аудио пустое")
                return None
            
            # Конвертируем в float32 для Gradio
            audio_numpy = audio.numpy().astype(np.float32)
            
            # Нормализуем
            if np.abs(audio_numpy).max() > 1.0:
                audio_numpy = audio_numpy / 32768.0
            
            # Проверяем что после нормализации не стало NaN
            if np.isnan(audio_numpy).any():
                print("⚠ Обнаружены NaN значения в аудио")
                return None
            
            duration = len(audio_numpy) / 24000
            print(f"✓ Готово: {len(audio_numpy)} сэмплов, {duration:.1f} сек")
            
            return (24000, audio_numpy)
            
        except Exception as e:
            print(f"⚠ Ошибка синтеза: {e}")
            
            # Если текст слишком длинный, пробуем еще сильнее обрезать
            if len(text) > 200:
                print("🔄 Пробуем с более коротким текстом...")
                return self.synthesize(text[:200], speaker, max_length=200)
            
            # Если все еще ошибка, возвращаем заглушку
            print("🔄 Используем заглушку...")
            return self._synthesize_fallback(text[:100])
        
    def _synthesize_fallback(self, text):
        """Запасной метод синтеза для очень коротких текстов"""
        try:
            audio = self.tts_model.apply_tts(
                text=text,
                speaker=self.current_speaker,
                sample_rate=24000
            )
            audio_numpy = audio.numpy().astype(np.float32)
            if np.abs(audio_numpy).max() > 1.0:
                audio_numpy = audio_numpy / 32768.0
            return (24000, audio_numpy)
        except:
            return None
            
    # def synthesize(self, text, speaker=None, max_length=500):
    #     """
    #     Синтез речи из текста с ограничением длины.
        
    #     Args:
    #         text: текст для озвучивания
    #         speaker: голос (baya, kseniya, xenia)
    #         max_length: максимальная длина текста (уменьшил до 500)
            
    #     Returns:
    #         tuple (sample_rate, audio_data) или None
    #     """
    #     if self.tts_model is None:
    #         print("⚠ TTS модель не загружена")
    #         return None
        
    #     if speaker is None:
    #         speaker = self.current_speaker
        
    #     # Ограничиваем длину текста
    #     if len(text) > max_length:
    #         # Ищем конец предложения для более естественного обрезания
    #         truncated = text[:max_length]
    #         last_period = truncated.rfind('.')
    #         last_question = truncated.rfind('?')
    #         last_exclamation = truncated.rfind('!')
            
    #         cut_point = max(last_period, last_question, last_exclamation)
    #         if cut_point > max_length * 0.7:  # Если нашли знак препинания в последней трети
    #             text = text[:cut_point + 1]
    #         else:
    #             text = truncated + '...'
            
    #         print(f"⚠ Текст обрезан с {len(text)} до {max_length} символов")
        
    #     print(f"🔊 Синтез речи ({len(text)} символов)...")
    #     try:
    #         # Пробуем синтезировать
    #         audio = self.tts_model.apply_tts(
    #             text=text,
    #             speaker=speaker,
    #             sample_rate=24000
    #         )
            
    #         # Конвертируем в float32 для Gradio
    #         audio_numpy = audio.numpy().astype(np.float32)
            
    #         # Нормализуем
    #         if np.abs(audio_numpy).max() > 1.0:
    #             audio_numpy = audio_numpy / 32768.0
            
    #         print(f"✓ Готово: {len(audio_numpy)} сэмплов, {len(audio_numpy)/24000:.1f} сек")
    #         return (24000, audio_numpy)
            
    #     except Exception as e:
    #         print(f"⚠ Ошибка синтеза: {e}")
            
    #         # Если текст все еще слишком длинный, пробуем еще сильнее обрезать
    #         if len(text) > 300:
    #             print("🔄 Пробуем с более коротким текстом...")
    #             return self.synthesize(text[:300], speaker, max_length=300)
            
    #         return None
    
    def test_microphone(self):
        """Тест микрофона с правильной частотой"""
        print("\n" + "="*50)
        print("🎤 ТЕСТ МИКРОФОНА")
        print("="*50)
        
        # Записываем сразу на нужной частоте
        recording = self.record_audio(duration=3)
        
        # Оборачиваем в формат как из интерфейса
        audio_input = (self.target_sample_rate, recording)
        
        # Пробуем распознать
        text = self.transcribe(audio_input)
        
        if text:
            print(f"\n✅ ТЕСТ УСПЕШЕН: '{text}'")
            return True
        else:
            print("\n❌ ТЕСТ НЕ УДАЛСЯ")
            return False