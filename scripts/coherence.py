import re
import numpy as np
import pandas as pd
import pymorphy3
from gensim.models import KeyedVectors
from gensim.downloader import load

morph = pymorphy3.MorphAnalyzer()

RUSSIAN_STOPWORDS = {'и', 'в', 'не', 'что', 'он', 'на', 'я', 'с', 'как', 'а', 'то', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'по', 'бы', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'когда', 'даже', 'ну', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'до', 'вас', 'там', 'потом', 'себя', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'без', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'тогда', 'кто', 'этот', 'потому', 'этого', 'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'много', 'эту', 'моя', 'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно'}

def load_model(path):
    return KeyedVectors.load_word2vec_format(path, binary=path.endswith('.bin'))

def cosine(v1, v2):
    if v1 is None or v2 is None:
        return 0.0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def lemmatize_word(word):
    """Приводит слово к нормальной форме (лемме)"""
    return morph.parse(word)[0].normal_form

def mean_sim(words, model):
    # words уже должны быть леммами
    vecs = [model[w] for w in words if w in model]
    if len(vecs) < 2:
        return 0.0
    sims = [cosine(vecs[i], vecs[j]) for i in range(len(vecs)) for j in range(i+1, len(vecs))]
    return np.mean(sims)

def linear_semantic_coherence(text, model, window=5):
    # 1. Токенизация и лемматизация
    tokens = re.findall(r'[а-яё]+', text.lower())
    # Лемматизируем ВСЕ слова, чтобы у модели были нормальные формы
    lemmas = [lemmatize_word(t) for t in tokens if t not in RUSSIAN_STOPWORDS and len(t) > 1]
    
    # 2. Находим индексы глаголов (уже среди лемм)
    verb_indices = []
    for i, lemma in enumerate(lemmas):
        pos = morph.parse(lemma)[0].tag.POS
        if pos in ('VERB', 'INFN'):
            verb_indices.append(i)
    
    if not verb_indices:
        return 0.0
    
    diffs = []
    for idx in verb_indices:
        start = max(0, idx - window)
        end = min(len(lemmas), idx + window + 1)
        window_lemmas = lemmas[start:end]
        verb_pos_in_window = idx - start
        
        # Coh_win (с глаголом)
        with_verb = mean_sim(window_lemmas, model)
        
        # CohV_win (без глагола)
        without_verb = mean_sim([w for i, w in enumerate(window_lemmas) if i != verb_pos_in_window], model)
        
        # CohDiff
        diffs.append(with_verb - without_verb)
    
    return np.mean(diffs)

# Использование
df = pd.read_csv("data/all_generated_metrics.csv")
model = load('glove-twitter-25')

df['coh_diff'] = df['text'].apply(lambda x: linear_semantic_coherence(x, model))
print(df['coh_diff'])
df.to_csv("data/all_generated_metrics.csv")