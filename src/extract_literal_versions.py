import pandas as pd
from dotenv import load_dotenv
from inference.processor import CheckpointProcessor

load_dotenv()

data = pd.read_csv("data/dataset_maxi.csv")
data_literature = data[data["text_domen"] == "художественный"]
data_news = data[data["text_domen"] == "новостной"]

# извлечение буквальных художественных текстов
literature_processor = CheckpointProcessor(
    model_name="GigaChat-2-Max",
    provider="gigachat",
    prompt_name="literal_v6",
    checkpoint_path="checkpoints/checkpoint_literal.csv",
    result_column_name="literal_version"
)

literature_literal = literature_processor.process(
    df=data_literature[["original_text", "idiomatic_expressions"]],
    columns_mapping={"original_text": "text", "idiomatic_expressions": "idioms"}
)

# извлечение буквальных новостных текстов
news_processor = CheckpointProcessor(
    model_name="GigaChat-2-Pro",
    provider="gigachat",
    prompt_name="literal_v6",
    checkpoint_path="checkpoints/checkpoint_news.csv",
    result_column_name="literal_version"
)

news_literal = news_processor.process(
    df=data_news[["original_text", "idiomatic_expressions"]],
    columns_mapping={"original_text": "text", "idiomatic_expressions": "idioms"}
)

# мерджинг и сохранение
all_literal_versions = pd.concat([literature_literal, news_literal], ignore_index=True)
all_literal_versions.to_csv("data/literal.csv", index=False)

# добавляем literal_version обратно в исходный датафрейм
data_literal = data.copy()
data_literal.loc[data_literal["text_domen"] == "художественный", "literal_version"] = literature_literal.values
data_literal.loc[data_literal["text_domen"] == "новостной", "literal_version"] = news_literal.values

data_literal.to_csv("data/dataset_maxi_literal.csv", index=False)
