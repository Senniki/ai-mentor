# Создайте временный файл test_silero.py для диагностики:
import torch

# Проверяем версию
print(f"Torch version: {torch.__version__}")

# Загружаем модель с явным указанием
model, _ = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_tts',
    language='ru',
    speaker='v4_ru',
    force_reload=True  # Принудительная перезагрузка
)

# Тестируем
audio = model.apply_tts(
    text="тест",
    speaker='baya',
    sample_rate=24000
)
print(f"Тест успешен, длина аудио: {len(audio)}")