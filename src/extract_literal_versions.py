import os
import pandas as pd
from dotenv import load_dotenv

from langfuse.langchain import CallbackHandler
from inference.model_constructor import ModelConstructor
from inference.model_inference import PromptConstructor, ModelInference

load_dotenv()

client = ModelConstructor.create_client("GigaChat-2", "gigachat")
inference = ModelInference(client)

prompt_builder = PromptConstructor("literal_v5")

data = pd.read_csv("data/dataset_maxi.csv")
texts_with_prompts = prompt_builder.render_from_df(
    data[["original_text", "idiomatic_expressions"]], 
    columns_mapping={"original_text": "text", "idiomatic_expressions": "idioms"}
    )

print(inference(texts_with_prompts[0]))