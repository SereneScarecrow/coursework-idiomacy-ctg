import re
import json
from pathlib import Path
import pymorphy3
import pandas as pd

# Русские стоп-слова (основные)
RUSSIAN_STOPWORDS = {
    'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой', 'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много', 'разве', 'эту', 'моя', 'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между'
}

def load_abstractness_dict(dict_path='evaluation/data/concretness-abstractness.json'):
    """Загружает словарь абстрактности."""
    with open(dict_path, encoding='utf-8') as f:
        return json.load(f)

def measure_abstractness(text: str, abstractness_dict: dict) -> float:
    """
    Вычисляет среднюю абстрактность текста.
    """
    morph = pymorphy3.MorphAnalyzer()
    
    # Токенизация
    words = re.findall(r'[а-яёА-ЯЁ]+', text.lower())
    
    # Фильтрация и лемматизация
    lemmas = []
    for word in words:
        if word not in RUSSIAN_STOPWORDS:
            lemma = morph.parse(word)[0].normal_form
            lemmas.append(lemma)
    
    if not lemmas:
        return 0.0
    
    # Поиск значений в словаре
    values = []
    for lemma in lemmas:
        if lemma in abstractness_dict:
            values.append(abstractness_dict[lemma])
    
    return sum(values) / len(values) if values else None

# Использование

dict_path = Path('data/concretness-abstractness.json')
if dict_path.exists():
    abstractness_dict = load_abstractness_dict(dict_path)

df = pd.read_csv("data/dataset_maxi_metrics.csv")

# Правильный способ - обернуть в lambda
df['abstractness_score'] = df['original_text'].apply(lambda x: measure_abstractness(x, abstractness_dict))


df.to_csv('data/dataset_maxi_metrics.csv')