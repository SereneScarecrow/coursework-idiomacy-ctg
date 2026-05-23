import re
import pandas as pd

def clean_text(text):
    """
    Очищает текст от переносов строк и восстанавливает разорванные слова.
    """
    if not isinstance(text, str):
        return text
    
    # восстанавливаем слова, разорванные переносом строки
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # заменяем все переносы строк на пробелы
    # text = text.replace('\r\n', ' ')
    # text = text.replace('\n', ' ')
    # text = text.replace('\r', ' ')
    
    # убираем множественные пробелы
    text = re.sub(r' +', ' ', text)
    
    # убираем пробелы перед знаками препинания
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    
    return text.strip()

data = pd.read_csv("data/dataset_maxi_literal(with_hyphens).csv")
data["original_text_clean"] = data["original_text"].apply(clean_text)

data.to_csv("data/dataset_maxi_literal.csv")