import torch
import numpy as np
from src.metrics.calculate_eer import compute_eer
from src.metrics.eer_metrics import EERMetric # Импортируй свой класс EERMetric

# Заглушка для функции compute_eer, чтобы не зависеть от её реализации
# (если она не доступна в скрипте)
def compute_eer_mock(bonafide_scores, other_scores):
    # Здесь можно вернуть любые тестовые значения
    return 0.1, 0.5 # Пример EER и порога

# Создаем тестовые данные
# 10 bona fide (метки 0) и 10 spoof (метки 1)
test_labels = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

# Скоры, которые будет выдавать модель
# Высокие скоры для bona fide, низкие для spoof
test_logits = torch.tensor([0.9, 0.85, 0.95, 0.88, 0.7, 0.6, 0.8, 0.9, 0.85, 0.95,
                            0.1, 0.2, 0.15, 0.05, 0.3, 0.25, 0.3, 0.1, 0.2, 0.05])

# Разделяем данные на батчи для имитации реального процесса
# Первый батч: 5 bona fide, 5 spoof
batch1_logits = test_logits[:10]
batch1_labels = test_labels[:10]

# Второй батч: 5 bona fide, 5 spoof
batch2_logits = test_logits[10:]
batch2_labels = test_labels[10:]

# 1. Инициализируем метрику
eer_metric = EERMetric()

# 2. Имитируем цикл по батчам и вызываем __call__
print("Накапливаем скоры из батча 1...")
eer_metric(batch1_logits, batch1_labels)
print(f"bonafide_scores после батча 1: {eer_metric.bonafide_scores}")
print(f"other_scores после батча 1: {eer_metric.other_scores}")

print("\nНакапливаем скоры из батча 2...")
eer_metric(batch2_logits, batch2_labels)
print(f"bonafide_scores после батча 2: {eer_metric.bonafide_scores}")
print(f"other_scores после батча 2: {eer_metric.other_scores}")

# 3. Вызываем result() для получения финального EER
final_eer = eer_metric.result()

print(f"\nФинальное значение EER: {final_eer}")
print(f"Ожидаемый EER будет около 0.0, так как скоры хорошо разделены")