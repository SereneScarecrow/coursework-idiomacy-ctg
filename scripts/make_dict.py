import pandas as pd
import json

def convert_xlsx_to_dict(input_file, output_file):
    """Конвертирует xlsx в JSON словарь."""
    df = pd.read_excel(input_file)
    
    # Берем первые две колонки
    words = df.iloc[:, 0].astype(str).str.lower()
    scores = df.iloc[:, 1]
    
    # Создаем словарь
    result = {}
    for word, score in zip(words, scores):
        if pd.notna(score):
            result[word] = float(score)
    
    # Сохраняем
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"Сохранено {len(result)} слов в {output_file}")

# Использование
convert_xlsx_to_dict('data/rejtingi.s.BERT.22.tysyach.slov.xlsx', 
                     'data/concretness-abstractness.json')