# src/web_ui.py
import gradio as gr
import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory.vector_store import VectorStore
from src.tools.mentor_tools import MentorTools
from src.core.mentor_agent import MentorAgent
from src.voice.voice_interface import VoiceInterface

class MentorApp:
    def __init__(self):
        print("=" * 50)
        print("Инициализация AI Наставника")
        print("=" * 50)
        
        # Корневая директория проекта
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Инициализация векторной БД
        print("\n[1/4] Загрузка базы знаний...")
        self.vs = VectorStore(
            persist_directory=os.path.join(self.root_dir, "chroma_db"),
            knowledge_base_path=os.path.join(self.root_dir, "knowledge_base")
        )
        self.retriever = self.vs.get_retriever(k=4)
        
        # Инициализация инструментов
        print("[2/4] Создание инструментов...")
        self.mentor_tools = MentorTools(self.retriever)
        self.tools = self.mentor_tools.get_all_tools()
        
        # Инициализация агента
        print("[3/4] Запуск агента...")
        self.agent = MentorAgent(self.tools)
        
        # Инициализация голоса
        print("[4/4] Загрузка голосового интерфейса...")
        self.voice = VoiceInterface()
        
        print("\n" + "=" * 50)
        print("✓ AI Наставник готов к работе!")
        print("=" * 50)
        print(f"📚 База знаний: {self.vs.knowledge_base_path}")
        print(f"🛠️ Инструментов: {len(self.tools)}")
        print(f"🎤 Голосовой ввод: {'✅' if self.voice.stt_model else '❌'}")
        print(f"🔊 Голосовой вывод: {'✅' if self.voice.tts_model else '❌'}")
        print("=" * 50 + "\n")
    
    # def process_message(self, message, history, use_voice=False):
    #     """Обработка текстового сообщения"""
    #     if not message:
    #         return "", history, None
        
    #     print(f"\n👤: {message}")
        
    #     # Получаем ответ от агента
    #     response = self.agent.invoke(message)
    #     print(f"🤖: {response[:200]}...")
        
    #     # Обновляем историю
    #     if history is None:
    #         history = []
    #     history.append({"role": "user", "content": message})
    #     history.append({"role": "assistant", "content": response})
        
    #     # Синтез речи если нужно - возвращаем ТОЛЬКО аудио, не историю
    #     audio = None
    #     if use_voice and self.voice.tts_model:
    #         audio_result = self.voice.synthesize(response)
    #         if audio_result:
    #             sample_rate, audio_data = audio_result
    #             audio = (int(sample_rate), audio_data)
    #             print(f"🔊 Аудио создано: {sample_rate}Hz, форма: {audio_data.shape}")
        
    #     # Важно: возвращаем 3 значения - пустой текст, обновленную историю, аудио
    #     return "", history, audio

    def process_message(self, message, history, use_voice=False):
        """Обработка текстового сообщения"""
        if not message:
            return "", history, None
        
        print(f"\n👤: {message}")
        
        # Получаем ответ от агента
        response = self.agent.invoke(message)
        print(f"🤖: {response[:200]}...")
        
        # Обновляем историю
        if history is None:
            history = []
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        
        # Синтез речи если нужно
        audio = None
        if use_voice and self.voice.tts_model and response and len(response.strip()) > 0:
            print(f"\n🔊 Подготовка голосового ответа...")
            
            # Если ответ слишком длинный, берем только первую часть
            if len(response) > 500:
                # Ищем конец первого предложения
                first_sentence = response.split('.')[0] + '.'
                if len(first_sentence) > 50:  # Минимальная длина
                    text_for_tts = first_sentence
                    print(f"⚠ Использую только первое предложение ({len(text_for_tts)} символов)")
                else:
                    text_for_tts = response[:500]
            else:
                text_for_tts = response
            
            audio = self.voice.synthesize(text_for_tts)
            audio = self.safe_audio_return(audio)
            
            if audio:
                print(f"✅ Аудио создано и проверено")
            else:
                print(f"❌ Не удалось создать аудио")
        
        return ("", history, audio)
    
    def process_voice_input(self, audio, history, use_voice=False):
        """Обработка голосового ввода"""
        if audio is None:
            return history, None
        
        print("\n" + "="*50)
        print("🎤 ГОЛОСОВОЙ ВВОД")
        print("="*50)
        
        # Распознаем речь
        text = self.voice.transcribe(audio)
        
        if text:
            print(f"\n✅ РАСПОЗНАНО: '{text}'")
            # ВАЖНО: process_message возвращает (str, list, tuple)
            # Но для voice_input нам нужно только (list, tuple)
            _, new_history, audio_output = self.process_message(text, history, use_voice)
            print(f"📤 Голосовой ввод возвращает: история({type(new_history)}), аудио({type(audio_output)})")
            return new_history, audio_output
        else:
            print("\n❌ НЕ РАСПОЗНАНО")
            gr.Warning("Речь не распознана")
            return history, None

    # def process_voice_input(self, audio, history, use_voice=False):
    #     """Обработка голосового ввода"""
    #     if audio is None:
    #         print("⚠ Нет аудиоданных")
    #         return history, None
        
    #     print(f"\n🎤 Получены аудиоданные: {type(audio)}")
        
    #     # Сохраняем аудио для отладки (опционально)
    #     if isinstance(audio, tuple) and len(audio) == 2:
    #         sample_rate, audio_data = audio
    #         print(f"   sample_rate: {sample_rate}")
    #         print(f"   форма данных: {audio_data.shape if hasattr(audio_data, 'shape') else 'N/A'}")
    #         print(f"   тип данных: {audio_data.dtype if hasattr(audio_data, 'dtype') else 'N/A'}")
    #         print(f"   диапазон: [{audio_data.min():.3f}, {audio_data.max():.3f}]")
        
    #     # Распознаем речь
    #     text = self.voice.transcribe(audio)
        
    #     if text:
    #         print(f"✅ Распознано: {text}")
    #         return self.process_message(text, history, use_voice)
    #     else:
    #         print("❌ Речь не распознана")
    #         gr.Warning("Речь не распознана. Попробуйте говорить четче или проверьте микрофон.")
    #         return history, None
    
    # def process_voice_input(self, audio, history, use_voice=False):
    #     """Обработка голосового ввода"""
    #     if audio is None:
    #         return history, None
        
    #     print(f"🎤 Получены аудиоданные: {type(audio)}")
        
    #     # Извлекаем аудиоданные
    #     audio_data = None
    #     if isinstance(audio, tuple) and len(audio) == 2:
    #         sample_rate, audio_data = audio
    #         print(f"   - Формат: кортеж, sample_rate={sample_rate}, форма данных={audio_data.shape if hasattr(audio_data, 'shape') else 'unknown'}")
    #     elif isinstance(audio, dict):
    #         # Новый формат Gradio
    #         audio_data = audio.get('data', audio.get('array'))
    #         print(f"   - Формат: словарь, ключи: {audio.keys()}")
    #     elif isinstance(audio, np.ndarray):
    #         audio_data = audio
    #         print(f"   - Формат: numpy array, форма={audio.shape}")
    #     else:
    #         audio_data = audio
    #         print(f"   - Формат: {type(audio)}")
        
    #     if audio_data is None:
    #         print("❌ Не удалось извлечь аудиоданные")
    #         gr.Warning("Ошибка обработки аудио")
    #         return history, None
        
    #     # Распознаем речь
    #     print("🔄 Распознавание речи...")
    #     text = self.voice.transcribe(audio_data)
        
    #     if text:
    #         print(f"📝 Распознано: {text}")
    #         # Передаем распознанный текст как обычное сообщение
    #         return self.process_message(text, history, use_voice)
    #     else:
    #         print("❌ Речь не распознана")
    #         gr.Warning("Речь не распознана. Попробуйте еще раз.")
    #         return history, None

    def safe_audio_return(self, audio):
        """Безопасно возвращает аудио, обрабатывая возможные ошибки"""
        if audio is None:
            return None
        
        try:
            # Проверяем что аудио в правильном формате
            if isinstance(audio, tuple) and len(audio) == 2:
                sr, data = audio
                if isinstance(data, np.ndarray) and data.size > 0:
                    return audio
        except Exception as e:
            print(f"⚠ Ошибка при проверке аудио: {e}")
        
        return None
    
    def create_ui(self):
        """Создает Gradio интерфейс"""
        
        with gr.Blocks(title="AI Mentor", theme=gr.themes.Soft()) as demo:
            gr.Markdown("""
            # 🤖 Инженерный AI Наставник
            ### Помощник в изучении робототехники, компьютерного зрения, тестирования и программирования
            """)
            
            with gr.Row():
                with gr.Column(scale=3):
                    # Чат
                    chatbot = gr.Chatbot(
                        label="Диалог",
                        height=500
                    )
                    
                    # Текстовый ввод
                    with gr.Row():
                        with gr.Column(scale=4):
                            msg = gr.Textbox(
                                label="Ваше сообщение",
                                placeholder="Спросите что-нибудь...",
                                show_label=False
                            )
                        with gr.Column(scale=1):
                            submit = gr.Button("📤 Отправить", variant="primary")
                    
                    # Голосовой ввод
                    voice_input = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        label="🎤 Голосовой ввод (нажмите и говорите)"
                    )
                    
                    # Голосовой вывод
                    voice_output = gr.Audio(
                        label="🔊 Голосовой ответ",
                        type="numpy",
                        autoplay=True,
                        visible=True
                    )
                    
                    # Кнопки управления
                    with gr.Row():
                        use_voice_cb = gr.Checkbox(
                            label="🔊 Озвучивать ответы",
                            value=True
                        )
                        clear_btn = gr.Button("🗑 Очистить диалог")
                        test_mic_btn = gr.Button("🎤 Тест микрофона")
                
                # Боковая панель с информацией
                with gr.Column(scale=1):
                    gr.Markdown("""
                    ### 📚 Темы
                    - **Робототехника**: ROS, управление, навигация
                    - **Компьютерное зрение**: OpenCV, детекция, трекинг
                    - **Тестирование**: unit tests, интеграция, автоматизация
                    - **Программирование**: Python, C++, алгоритмы
                    
                    ### 💡 Примеры вопросов
                    - "Составь план изучения OpenCV"
                    - "Объясни топики в ROS"
                    - "Дай задание по unit testing"
                    - "Что такое фильтрация изображений?"
                    
                    ### 🎯 Возможности
                    - Текстовый и голосовой ввод
                    - Озвучивание ответов
                    - Поиск в базе знаний
                    - Поиск в интернете
                    - Выполнение Python кода
                    
                    ### ⚙️ Статус
                    """)
                    
                    # Индикаторы статуса
                    status_text = gr.Markdown(f"""
                    - 🎤 Микрофон: {'✅' if self.voice.stt_model else '❌'}
                    - 🔊 Динамики: {'✅' if self.voice.tts_model else '❌'}
                    - 📚 База знаний: {'✅' if os.path.exists(self.vs.knowledge_base_path) else '❌'}
                    """)
            
            # Обработчики событий
            msg.submit(
                self.process_message,
                inputs=[msg, chatbot, use_voice_cb],
                outputs=[msg, chatbot, voice_output]
            )
            
            submit.click(
                self.process_message,
                inputs=[msg, chatbot, use_voice_cb],
                outputs=[msg, chatbot, voice_output]
            )
            
            # Для аудио используем событие stop_recording
            voice_input.stop_recording(
                self.process_voice_input,
                inputs=[voice_input, chatbot, use_voice_cb],
                outputs=[chatbot, voice_output]
            )
            
            clear_btn.click(
                lambda: ([], None, self.agent.clear_memory()),
                outputs=[chatbot, voice_output, msg],
                queue=False
            )
            
            test_mic_btn.click(
                self.voice.test_microphone,
                outputs=[],
                queue=False
            ).then(
                lambda: gr.Info("Тест микрофона завершен"),
                None, None
            )
        
        return demo

if __name__ == "__main__":
    app = MentorApp()
    demo = app.create_ui()
    
    print("\n🚀 Запуск веб-интерфейса...")
    print("   Локальный адрес: http://127.0.0.1:7860")
    print("   Для остановки нажмите Ctrl+C\n")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )