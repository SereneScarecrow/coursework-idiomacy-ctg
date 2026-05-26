from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

from evaluation.evaluator import QualityEvaluator, IdiomEvaluator
from inference.processor import CheckpointProcessor
from inference.model_constructor import FormatJudge

load_dotenv()

data = pd.read_csv("data/dataset_maxi_literal.csv")

original_texts = data["original_text_clean"]

judge_processor = CheckpointProcessor(
    model_name="GigaChat-2-Max",
    provider="gigachat",
    prompt_name="judge_prompt",
    structured_output_schema=FormatJudge,
    checkpoint_path="checkpoints/judge_results.csv",
    temperature=0.3
)

judge_results = judge_processor.process(df=original_texts)

final_results = pd.DataFrame({
    'original_text': original_texts,
    'judge_grammar': judge_results['grammar'],
    'judge_fluency': judge_results['fluency'],
    'judge_naturalness': judge_results['naturalness'],
    'judge_idiomaticity': judge_results['idiomaticity'],
    'judge_idioms': judge_results['idiomatic_expressions']
})

final_results.to_csv("data/validate_judje.csv", index=False)