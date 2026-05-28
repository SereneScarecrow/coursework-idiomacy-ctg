from pathlib import Path
import pandas as pd
from tqdm import tqdm

from evaluation.quality_evaluator import QualityEvaluator
from evaluation.model_loader import ModelConfig

def evaluate_quality(
    df: pd.DataFrame,
    reference_col: str = 'original_text',
    candidate_col: str = 'literal_version',
    metrics: list = ['wmd'],
    device: str = 'cuda',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Рассчитывает метрики качества для каждой пары текстов.
    
    Параметры:
        df: датафрейм с колонками original_text и literal_version
        reference_col: колонка с оригинальными текстами
        candidate_col: колонка с буквальными версиями
        metrics: список метрик для расчета
        device: 'cuda' или 'cpu'
        verbose: показывать прогресс
    
    Возвращает:
        Исходный датафрейм с добавленными колонками метрик
    """
    # Инициализация оценщика
    evaluator = QualityEvaluator(gpu=(device == 'cuda'), verbose=verbose, config=ModelConfig())
    
    # Подготовка данных (удаляем пустые строки)
    valid_mask = df[reference_col].notna() & df[candidate_col].notna()
    valid_mask &= df[reference_col].astype(str).str.strip().ne('') & df[candidate_col].astype(str).str.strip().ne('')
    
    if verbose:
        print(f"Оценка {valid_mask.sum()} из {len(df)} пар текстов...")
    
    # Расчет метрик только для валидных строк
    results_df = evaluator.run_evaluation_pipeline(
        references=df.loc[valid_mask, reference_col].tolist(),
        candidates=df.loc[valid_mask, candidate_col].tolist(),
        metrics=metrics
    )
    
    # Добавление результатов в исходный датафрейм
    for col in results_df.columns:
        df[col] = float('nan')
        df.loc[valid_mask, col] = results_df[col].values
    
    # Вывод средних значений
    if verbose:
        print("\nСредние значения метрик:")
        for col in results_df.columns:
            print(f"  {col}: {df[col].mean():.4f}")
    
    return df

# Загрузка вашего датафрейма
df = pd.read_csv('data/dataset_maxi_judge.csv')

# Расчет метрик
result_df = evaluate_quality(df, reference_col='original_text', candidate_col='literal_version')

# Сохранение результата
result_df.to_csv('data/dataset_maxi_with_metrcis.csv', index=False)

# -----------------------------------------------------------
# Загрузка датафреймов
dataset_maxi_literal = pd.read_csv('data/dataset_maxi_literal.csv')
all_generated = pd.read_csv('data/all_generated_judge.csv')

# Создаем датафрейм для оценки, соединяя по индексу
# У all_generated есть колонка 'index_orig', которая ссылается на индекс dataset_maxi_literal
df_for_evaluation = all_generated.copy()

# Добавляем колонку с буквальными текстами из dataset_maxi_literal по индексу
df_for_evaluation['literal_version'] = df_for_evaluation['index_orig'].map(dataset_maxi_literal['literal_version'])

# Добавляем оригинальные тексты (если нужны как референс)
df_for_evaluation['original_text'] = df_for_evaluation['index_orig'].map(dataset_maxi_literal['original_text'])

# Проверяем, что маппинг прошел успешно
print(f"Успешно сопоставлено: {df_for_evaluation['literal_version'].notna().sum()} из {len(df_for_evaluation)} строк")

# Расчет метрик: сравниваем 'text' (сгенерированный) с 'literal_version' (эталон)
# Примечание: в функции evaluate_quality:
#   - reference_col = 'literal_version' (то, с чем сравниваем, эталон)
#   - candidate_col = 'text' (то, что оцениваем, сгенерированный текст)
all_generated_metrics = evaluate_quality(
    df=df_for_evaluation,
    reference_col='literal_version',  # эталон из dataset_maxi_literal
    candidate_col='text',              # сгенерированный текст из all_generated
    metrics=['wmd'],
    device='cuda',
    verbose=True
)

# Сохраняем результат
all_generated_metrics.to_csv('data/all_generated_metrics.csv', index=False)
print(f"\nРезультаты сохранены в 'data/all_generated_with_metrics.csv'")
print(f"Добавлены колонки: {[col for col in all_generated_metrics.columns if col.endswith('_score')]}")


