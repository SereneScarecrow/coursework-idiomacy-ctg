import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

from langfuse.langchain import CallbackHandler
from inference.model_constructor import ModelConstructor
from inference.model_inference import PromptConstructor, ModelInference

load_dotenv()

def extract_literal_versions(model_name, provider, prompt, df, columns_mapping,
                             checkpoint_path="checkpoint.csv", resume=True):
    """
    Функция для извлечения буквальных версий текстов с чекпоинтами.
    """
    # инициализация клиента
    inference = ModelInference(
        ModelConstructor.create_client(model_name, provider)
        )
    # загрузка промпта
    prompt_builder = PromptConstructor(prompt)
    # рендер промптов с переменными
    texts_with_prompts = prompt_builder.render_from_df(df, columns_mapping=columns_mapping)

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

data = pd.read_csv("data/dataset_maxi.csv")
data_literature = data[data["text_domen"] == "художественный"]
data_news = data[data["text_domen"] == "новостной"]

# извлечение буквальных художественных текстов
literature_literal = extract_literal_versions(
    model_name="GigaChat-2-Max",
    provider="gigachat",
    prompt="literal_v6",
    df=data_literature[["original_text", "idiomatic_expressions"]],
    columns_mapping={"original_text": "text", "idiomatic_expressions": "idioms"},
    checkpoint_path="checkpoints/checkpoint_literal.csv")

# извлечение буквальных новостных текстов
news_literal = extract_literal_versions(
    model_name="GigaChat-2-Pro",
    provider="gigachat",
    prompt="literal_v6",
    df=data_news[["original_text", "idiomatic_expressions"]],
    columns_mapping={"original_text": "text", "idiomatic_expressions": "idioms"},
    checkpoint_path="checkpoints/checkpoint_news.csv") 

# мерджинг и сохранение
all_literal_versions = pd.concat([literature_literal, news_literal], ignore_index=True)
all_literal_versions.to_csv("data/literal.csv", index=False)

data['literal_version'] = all_literal_versions
data.to_csv("data/dataset_maxi_literal.csv")
