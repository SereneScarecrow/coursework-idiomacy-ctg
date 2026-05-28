from pathlib import Path
import pandas as pd
import re
from dotenv import load_dotenv

from inference.processor import CheckpointProcessor
from inference.model_constructor import FormatJudge

load_dotenv()

data = pd.read_csv("data/dataset_maxi_literal.csv")

judge_processor = CheckpointProcessor(
    model_name="GigaChat-2-Max",
    provider="gigachat",
    prompt_name="judge_v1",
    structured_output_schema=FormatJudge,
    checkpoint_path="checkpoints/judge_results2.csv"
)

def unpack_judge_result(judge_scores):
    """Распаковывает judge_scores из строки в отдельные колонки"""
    if pd.isna(judge_scores):
        return pd.Series({
            'grammar': None,
            'fluency': None,
            'naturalness': None,
            'idiomaticity': None,
            'idiom_diversity': None,
            'idiomatic_expressions': ""
        })
    
    scores_str = str(judge_scores)
    
    # Парсим числовые поля
    grammar_match = re.search(r'grammar=(\d+)', scores_str)
    fluency_match = re.search(r'fluency=(\d+)', scores_str)
    naturalness_match = re.search(r'naturalness=(\d+)', scores_str)
    idiom_diversity_match = re.search(r'idiom_diversity=(\d+)', scores_str)
    idiomaticity_match = re.search(r'idiomaticity=(\d+)', scores_str)
    
    # Извлекаем idiomatic_expressions
    expressions_match = re.search(r'idiomatic_expressions=(.+)', scores_str)
    expressions_value = expressions_match.group(1).strip() if expressions_match else ""
    
    return pd.Series({
        'grammar': int(grammar_match.group(1)) if grammar_match else None,
        'fluency': int(fluency_match.group(1)) if fluency_match else None,
        'naturalness': int(naturalness_match.group(1)) if naturalness_match else None,
        'idiomaticity': int(idiomaticity_match.group(1)) if idiomaticity_match else None,
        'idiom_diversity': int(idiom_diversity_match.group(1)) if idiomaticity_match else None,
        'idiomatic_expressions': expressions_value
    })

# Получаем результаты от judge_processor
result_df = judge_processor.process(data["original_text"])

# Если result_df содержит колонку 'result' со строковыми значениями
if 'result' in result_df.columns:
    # Распаковываем каждую строку в отдельные колонки
    unpacked = result_df['result'].apply(unpack_judge_result)
    # Присоединяем распакованные колонки к исходным данным
    data = pd.concat([data, unpacked], axis=1)
else:
    # Если result_df уже содержит распакованные колонки
    data = pd.concat([data, result_df], axis=1)

data.to_csv("data/dataset_maxi_judge.csv", index=False)