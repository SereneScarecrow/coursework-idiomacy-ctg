import pandas as pd
import re
from pathlib import Path

def merge_csv_with_metadata(input_dir, output_file='merged_data.csv'):
    """
    Объединяет все CSV файлы вида idiomatic_уровень_модель_размер.csv
    Добавляет колонки: level (2/1/0), model, size
    """
    input_path = Path(input_dir)
    all_data = []
    
    # Маппинг уровня в числовое значение
    level_map = {
        'high': 2,
        'medium': 1,
        'low': 0
    }
    
    for csv_file in input_path.glob('idiomatic_*.csv'):
        # Парсим имя файла: idiomatic_high_qwen2.5_7b.csv
        match = re.match(r'idiomatic_(\w+)_([^_]+(?:_[^_]+)*?)_(\d+(?:\.\d+)?[bB]?)\.csv', csv_file.name)
        
        if not match:
            print(f"Пропущен файл с неожиданным форматом: {csv_file.name}")
            continue
        
        level_raw, model, size = match.groups()
        level = level_map.get(level_raw.lower(), -1)
        
        # Читаем файл
        df = pd.read_csv(csv_file)
        
        # Добавляем мета-колонки
        df['level'] = level
        df['model'] = model
        df['size'] = size
        
        all_data.append(df)
    
    if not all_data:
        print("Не найдено ни одного подходящего файла")
        return None
    
    # Объединяем и сохраняем
    result = pd.concat(all_data, ignore_index=True)
    result.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Объединено {len(all_data)} файлов. Результат: {output_file}")
    
    return result


df = merge_csv_with_metadata('data/generated', 'data/all_generated.csv')