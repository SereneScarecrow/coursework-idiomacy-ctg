from typing import List
import pandas as pd
from tqdm import tqdm

from evaluation.idiom_evaluator import IdiomEvaluator

def evaluate_idioms(
    df: pd.DataFrame,
    text_col: str = 'text',
    original_col: str = 'original_text',
    literal_col: str = 'literal_version',
    idioms_col: str = 'idiomatic_expressions',
    metrics: List[str] = None,
    device: str = 'cuda',
    verbose: bool = True
) -> pd.DataFrame:
    """Оценка текстов с идиомами"""
    
    if metrics is None:
        metrics = ['perplexity', 'surprisal', 'abstractness', 'word_frequency', 'cosine_distances']
    
    evaluator = IdiomEvaluator(gpu=(device == 'cuda'), verbose=verbose)
    
    # Подготовка данных
    valid_mask = (df[text_col].notna() & df[original_col].notna() & 
                  df[literal_col].notna() & df[idioms_col].notna())
    
    if verbose:
        print(f"Оценка {valid_mask.sum()} из {len(df)} текстов...")
    
    # Обработка каждой строки
    results = []
    iterator = tqdm(df[valid_mask].iterrows(), total=valid_mask.sum(), desc="Processing") if verbose else df[valid_mask].iterrows()
    
    for _, row in iterator:
        # Парсим идиомы (если строка, превращаем в список)
        idioms = row[idioms_col]
        if isinstance(idioms, str):
            idioms = [i.strip() for i in idioms.replace(';', ',').split(',') if i.strip()]
        
        try:
            result = evaluator.run_evaluation_pipeline(
                text=row[text_col],
                original=row[original_col],
                literal=row[literal_col],
                idioms=idioms if isinstance(idioms, list) else [idioms],
                metrics=metrics
            )
            results.append(result.iloc[0].to_dict())
        except:
            results.append({f"{m}_score" if m != 'surprisal' else 'idiom_surprisal_score': float('nan') 
                          for m in metrics})
    
    # Добавляем результаты в датафрейм
    results_df = pd.DataFrame(results)
    for col in results_df.columns:
        df[col] = float('nan')
        df.loc[valid_mask, col] = results_df[col].values
    
    if verbose:
        print("\nСредние значения:")
        for col in results_df.select_dtypes(include=['float64']).columns:
            print(f"  {col}: {df[col].mean():.4f}")
    
    return df

df = pd.read_csv("data/all_generated_metrics.csv")[:5]
df = evaluate_idioms(df, metrics = ['abstractness', 'word_frequency', 'cosine_distances'])
print(df)