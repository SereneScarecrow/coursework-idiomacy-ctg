import re
from wordfreq import zipf_frequency
import pandas as pd
# Русские стоп-слова (те же самые)
RUSSIAN_STOPWORDS = {
    'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой', 'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много', 'разве', 'эту', 'моя', 'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между'
}

def measure_word_frequency(text: str) -> dict:
    """
    Возвращает среднюю частоту и долю известных слов.
    
    Returns:
        dict: {'mean_frequency': float, 'coverage': float}
    """
    # Токенизация
    words = re.findall(r'[а-яёА-ЯЁ]+', text.lower())
    
    # Фильтрация стоп-слов
    filtered = [word for word in words if word not in RUSSIAN_STOPWORDS and len(word) > 1]
    
    if not filtered:
        return {'mean_frequency': 0.0, 'coverage': 0.0}
    
    freqs = []
    known_count = 0
    
    for word in filtered:
        freq = zipf_frequency(word, 'ru')
        if freq > 0:
            freqs.append(freq)
            known_count += 1
    
    coverage = known_count / len(filtered) if filtered else 0.0
    
    return {
        'mean_frequency': sum(freqs) / len(freqs) if freqs else 0.0,
        'coverage': coverage
    }


# Использование с DataFrame
df = pd.read_csv("data/dataset_maxi_metrics.csv")

# Применяем функцию
df['word_freq_mean'] = df['original_text'].apply(lambda x: measure_word_frequency(x)['mean_frequency'])
df['word_freq_coverage'] = df['original_text'].apply(lambda x: measure_word_frequency(x)['coverage'])

print(df[['text', 'word_freq_mean', 'word_freq_coverage']])

df.to_csv("data/dataset_maxi_metrics.csv")