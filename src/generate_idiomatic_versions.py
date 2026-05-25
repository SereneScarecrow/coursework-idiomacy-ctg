import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

from langfuse.langchain import CallbackHandler
from inference.model_constructor import ModelConstructor
from inference.model_inference import FormatIdiomatic, PromptConstructor, ModelInference

load_dotenv()

def generate_idiomatic_versions(model_name, provider, prompt, df,
                             checkpoint_path="checkpoint.csv", resume=True):
    """
    Функция для извлечения буквальных версий текстов с чекпоинтами.
    """
    # инициализация клиента с логгированием
    langfuse_handler = CallbackHandler()

    inference = ModelInference(
        ModelConstructor.create_client(model_name, provider, structured_output_schema=FormatIdiomatic),
        langfuse_handler=langfuse_handler
        )
    # загрузка промпта
    prompt_builder = PromptConstructor(prompt)
    # рендер промптов с переменными
    texts_with_prompts = prompt_builder.render_from_df(df)

    # загрузка чекпоинта
    results = []
    processed_indices = set()
    
    if resume and os.path.exists(checkpoint_path):
        saved_df = pd.read_csv(checkpoint_path)
        if not saved_df.empty:
            for _, row in saved_df.iterrows():
                literal_val = row['literal_version']
                # проверяем на None/NaN и добавляем в processed_indices
                if pd.notna(literal_val) and literal_val is not None:
                    results.append(literal_val)
                    processed_indices.add(row['index'])
            print(f"Загружено {len(results)} валидных результатов из {len(saved_df)} сохраненных")
    
    # обработка
    for idx, text in tqdm(enumerate(texts_with_prompts), total=len(texts_with_prompts)):
        if idx in processed_indices:
            continue
            
        try:
            literal_text = inference(text)
            results.append(literal_text)
            
            # сохраняем чекпоинт
            checkpoint_data = {
                'index': list(range(len(results))),
                'literal_version': results
            }
            pd.DataFrame(checkpoint_data).to_csv(checkpoint_path, index=False)
            
        except Exception as e:
            print(f"Ошибка на индексе {idx}: {e}")
            # в случае ошибки сохраняем пустое значение
            results.append(None)
            
            checkpoint_data = {
                'index': list(range(len(results))),
                'literal_version': results
            }
            pd.DataFrame(checkpoint_data).to_csv(checkpoint_path, index=False)
    
    return pd.Series(results)

data = pd.read_csv("data/dataset_maxi_literal.csv").sample(n=5)

# извлечение буквальных художественных текстов
idiomatic_versions = generate_idiomatic_versions(
    model_name="qwen3:8b",
    provider="ollama",
    prompt="idiomatic_high",
    df=data["literal_version"],
    checkpoint_path="checkpoints/checkpoint_qwen3_14b_high.csv")

print(idiomatic_versions)