import os
os.environ['OLLAMA_HOST'] = '127.0.0.1:11434'

import pandas as pd
from dotenv import load_dotenv
from inference.processor import CheckpointProcessor
from inference.model_constructor import FormatIdiomatic

load_dotenv()

data = pd.read_csv("data/dataset_maxi_literal.csv")

# создаём процессор
processor = CheckpointProcessor(
    model_name="qwen2.5:3b",
    provider="ollama",
    prompt_name="idiomatic_high",
    structured_output_schema=FormatIdiomatic,
    checkpoint_path="checkpoints/checkpoint_qwen-2.5-3b-high-without_examples.csv",
    temperature=0.75,
    result_column_name="idiomatic_version"
)

# запускаем обработку
idiomatic_versions = processor.process(
    df=data["literal_version"],
)

idiomatic_versions.to_csv("data/idiomatic_high_qwen2.5_3b.csv")