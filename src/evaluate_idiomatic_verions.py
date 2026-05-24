from pathlib import Path
import pandas as pd
from evaluation.evaluator import QualityEvaluator

# Загрузка данных
data = pd.read_csv("data/dataset_maxi_literal.csv")

# Берём первые 5 примеров
original_texts = data["original_text_clean"][-5:].tolist()
literal_texts = data["literal_version"][-5:].tolist()

# Инициализация оценщика
evaluator = QualityEvaluator(gpu=False, verbose=True)

# Расчёт метрик
df_eval = evaluator.run_evaluation_pipeline(
    references=original_texts,
    candidates=literal_texts,
    metrics=['bleurt']
)

# Результаты
print("Оригинальные тексты:")
print(df_eval)
print("\nБуквальные тексты:")