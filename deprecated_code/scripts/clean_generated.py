import pandas as pd
from pathlib import Path

def process_csv_files(input_dir, output_dir='generated_clean', target_col='column_name'):
    """
    Разбивает колонку вида 'text=... ratio=...' на две отдельные колонки
    во всех CSV файлах указанной директории.
    """
    input_path = Path(input_dir)
    out_path = input_path / output_dir
    out_path.mkdir(exist_ok=True)

    csv_files = list(input_path.glob('*.csv'))
    if not csv_files:
        return

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        if target_col not in df.columns:
            continue

        # Разделяем text и ratio по шаблону
        df[['text', 'ratio']] = df[target_col].str.extract(r'text=\s*(.*?)\s*ratio=\s*(.*)$')

        # Если ничего не распарсилось — пропускаем файл
        if df['text'].isna().all():
            continue

        df = df.drop(columns=[target_col])
        df.to_csv(out_path / csv_file.name, index=False, encoding='utf-8-sig')


process_csv_files(
    input_dir='data/generation/',
    output_dir='generated_clean',
    target_col='idiomatic_version'
)