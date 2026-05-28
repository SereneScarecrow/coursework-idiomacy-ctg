import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import nltk
import numpy as np
import pandas as pd
import pymorphy3
import torch
import torch.nn.functional as F
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from wordfreq import zipf_frequency, tokenize
from transformers import AutoModelForSequenceClassification, BertTokenizerFast

from evaluation.evaluator import Evaluator
from evaluation.model_loader import ModelConfig, ModelLoader

@dataclass
class TargetSpan:
    """Представляет целевую конструкцию в тексте."""
    text: str           # Оригинальный текст конструкции
    start_char: int     # Начальная позиция в исходном тексте
    end_char: int       # Конечная позиция в исходном тексте
    tokens: List[str]   # Токены, составляющие конструкцию
    token_indices: List[int]  # Индексы токенов в полной токенизации

class IdiomEvaluator(Evaluator):
    def __init__(self, gpu: bool = True, verbose: bool = True, config: ModelConfig = None):
        super().__init__(verbose=verbose)
        self.device = 'cuda' if (gpu and torch.cuda.is_available()) else 'cpu'
        self.config = config or ModelConfig()
        self.loader = ModelLoader()

        self.max_length = getattr(self.config, 'max_length', 512)
        
        # Ленивая загрузка модели для перплексии/сюрпрайза
        self.ppl_model = None
        self.ppl_tokenizer = None
        self.embedding_model = None
        self.embedding_tokenizer = None

        # загрузка словаря для абстрактности-конкретности
        self.morph = pymorphy3.MorphAnalyzer()
        self.abstractness_dict = self._load_dict()

        # инициализация модели для анализа тональности
        self.sentiment_model = None
        self.sentiment_tokenizer = None
        self.sentiment_labels = None
    
    def _load_sentiment_model(self):
        """Ленивая загрузка модели для анализа тональности через ModelLoader."""
        if self.sentiment_model is None:
            model_name = self.config.sentiment_model
            
            if self.verbose:
                print(f"Loading sentiment model: {model_name}")
            
            self.sentiment_model, self.sentiment_tokenizer, self.sentiment_labels = \
                self.loader.load_sentiment_model(model_name, device=self.device)
            
            # Переопределяем метки, если они заданы в конфиге
            if self.config.sentiment_labels is not None:
                self.sentiment_labels = self.config.sentiment_labels
        
        return self.sentiment_model is not None
    
    def _load_dict(self) -> Dict[str, float]:
        path = 'data/concretness-abstractness.json'
        with open(path, encoding='utf-8') as f:
            print('found file')
            return json.load(f)
    
    def _load_perplexity_model(self):
        """Ленивая загрузка модели для расчета перплексии и сюрпрайза."""
        if self.ppl_model is None:
            self.ppl_model, self.ppl_tokenizer = self.loader.load_perplexity_model(
                self.config.perplexity_model, 
                device=self.device
            )
        return self.ppl_model is not None
    
    def _load_embedding_model(self):
        """Ленивая загрузка модели для получения эмбеддингов текстов."""
        if self.embedding_model is None:
            self.embedding_model, self.embedding_tokenizer = self.loader.load_embedding_model(
                self.config.embedding_model,
                device=self.device
            )
        return self.embedding_model is not None
    
    def _find_target_spans(self, text: str, target_phrases: List[str]) -> List[TargetSpan]:
        """
        Находит все вхождения целевых фраз в тексте.
        
        Args:
            text: Исходный текст
            target_phrases: Список фраз для поиска (метафоры, идиомы и т.д.)
        
        Returns:
            Список TargetSpan с информацией о позициях
        """
        spans = []
        
        for phrase in target_phrases:
            # Ищем все вхождения (регистронезависимо для русского)
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            
            for match in pattern.finditer(text):
                spans.append(TargetSpan(
                    text=match.group(),
                    start_char=match.start(),
                    end_char=match.end(),
                    tokens=[],
                    token_indices=[]
                ))
        
        # Сортируем по позиции в тексте
        spans.sort(key=lambda x: x.start_char)
        return spans
    
    def _tokenize_and_align(self, text: str, spans: List[TargetSpan]) -> List[TargetSpan]:
        """
        Токенизирует текст и выравнивает spans с токенами.
        
        Использует offset_mapping для точного определения,
        какие токены покрывают каждую целевую конструкцию.
        """
        if not self._load_perplexity_model():
            return spans
        
        # Токенизируем с возвратом смещений
        encodings = self.ppl_tokenizer(
            text,
            return_tensors='pt',
            return_offsets_mapping=True,
            add_special_tokens=True
        )
        
        offsets = encodings['offset_mapping'][0]  # (start, end) для каждого токена
        input_ids = encodings['input_ids'][0]
        
        # Для каждого span определяем, какие токены в него входят
        for span in spans:
            token_indices = []
            tokens = []
            
            for i, (start, end) in enumerate(offsets):
                # Проверяем перекрытие span'а с токеном
                if start < span.end_char and end > span.start_char:
                    token_indices.append(i)
                    # Декодируем токен обратно в строку
                    token = self.ppl_tokenizer.decode([input_ids[i]])
                    tokens.append(token)
            
            span.token_indices = token_indices
            span.tokens = tokens
        
        return spans
    
    def _compute_token_surprisal_sequence(self, text: str, token_indices: List[int]) -> List[float]:
        """
        Вычисляет сюрпрайз для последовательности токенов (для GPT).
        """
        if not self._load_perplexity_model():
            return [float('inf')] * len(token_indices)
        
        # Токенизируем один раз
        encodings = self.ppl_tokenizer(
            text,
            return_tensors='pt',
            add_special_tokens=True,
            padding=False,  # Не паддим, чтобы индексы не съехали
            truncation=True
        )
        
        input_ids = encodings['input_ids'].to(self.device)
        
        with torch.no_grad():
            outputs = self.ppl_model(input_ids)
            logits = outputs.logits  # [1, seq_len, vocab_size]
        
        surprisals = []
        
        for token_pos in token_indices:
            # Для GPT: предсказание для позиции pos находится в логитах позиции pos-1
            if token_pos == 0:
                # Первый токен не имеет предсказания (или предсказывается из BOS)
                # Пропускаем или используем специальную обработку
                if self.verbose:
                    print(f"[WARNING] First token (pos=0) has no prior context")
                continue
            
            # Берем логиты с предыдущей позиции
            logits_for_pred = logits[0, token_pos - 1, :]
            
            # Используем log_softmax (более эффективно)
            log_probs = torch.log_softmax(logits_for_pred, dim=-1)
            true_token_id = input_ids[0, token_pos]
            
            surprisal = -log_probs[true_token_id].item()
            surprisals.append(surprisal)
        
        return surprisals
    
    def measure_phrase_surprisal(self, text: str, target_phrases: List[str]) -> List[Tuple[str, float, List[float]]]:
        """
        Вычисляет сюрпрайз для выделенных фраз в тексте.
        
        Args:
            text: Исходный текст
            target_phrases: Список целевых фраз (метафор, идиом и т.д.)
        
        Returns:
            Список кортежей: (фраза, общий сюрпрайз, список сюрпрайзов по токенам)
        
        Пример:
            >>> evaluator = IdiomEvaluator()
            >>> text = "Он зарыл свой талант в землю и ничего не добился."
            >>> phrases = ["зарыл талант в землю"]
            >>> results = evaluator.measure_phrase_surprisal(text, phrases)
            >>> # results = [("зарыл талант в землю", 12.34, [2.1, 3.2, 4.5, 2.54])]
        """
        if not target_phrases:
            return []
        
        # 1. Находим все вхождения фраз в тексте
        spans = self._find_target_spans(text, target_phrases)
        
        if not spans:
            if self.verbose:
                print(f"[WARNING] No spans found for phrases: {target_phrases}")
            return [(phrase, float('inf'), []) for phrase in target_phrases]
        
        # 2. Выравниваем с токенами
        spans = self._tokenize_and_align(text, spans)
        
        # 3. Для каждого span вычисляем сюрпрайз
        results = []
        
        for span in spans:
            if not span.token_indices:
                # Фраза не найдена в токенизации (редко, но бывает)
                results.append((span.text, float('inf'), []))
                continue
            
            # Вычисляем сюрпрайз для последовательности токенов
            token_surprisals = self._compute_token_surprisal_sequence(text, span.token_indices)
            
            # нормируем для общего сюрпрайза фразы
            total_surprisal = sum(token_surprisals) / len(token_surprisals)
            
            results.append((span.text, total_surprisal, token_surprisals))
        
        return results
    
    def measure_perplexity(self, text: str) -> float:
        if not self._load_perplexity_model():
            return float('inf')
        
        encodings = self.ppl_tokenizer(text, return_tensors='pt', add_special_tokens=True)
        input_ids = encodings['input_ids'].to(self.device)
        
        with torch.no_grad():
            outputs = self.ppl_model(input_ids, labels=input_ids)
        
        return torch.exp(outputs.loss).item()
    
    def _get_text_embedding(self, text: str, strategy: str = 'mean') -> np.ndarray:
        """
        Получает эмбеддинг для текста.
        
        Args:
            text: Входной текст
            strategy: 'mean' (усреднение токенов) или 'cls' ([CLS] токен)
        
        Returns:
            Вектор размерности (embedding_dim,)
        """
        if not self._load_embedding_model():
            return np.zeros(768)
        
        inputs = self.embedding_tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            
            if strategy == 'cls':
                # Берем [CLS] токен (первый)
                text_embedding = outputs.last_hidden_state[:, 0, :]
            else:  # 'mean' по умолчанию
                # Усреднение токенов с учетом маски внимания
                token_embeddings = outputs.last_hidden_state
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                
                masked_embeddings = token_embeddings * attention_mask
                sum_embeddings = masked_embeddings.sum(dim=1)
                mask_sum = attention_mask.sum(dim=1)
                text_embedding = sum_embeddings / mask_sum
        
        return text_embedding.squeeze().cpu().numpy()
    
    def measure_pairwise_cosine_distance(
        self, 
        text1: str, 
        text2: str, 
        text3: str,
        strategy: str = 'mean'
    ) -> Dict[Tuple[str, str], float]:
        """
        Вычисляет попарные косинусные расстояния между тремя текстами.
        
        Args:
            text1, text2, text3: Три текста для сравнения
            strategy: Способ получения эмбеддинга ('mean' или 'cls')
        
        Returns:
            Словарь с ключами-кортежами (text1, text2) и значениями-расстояниями.
        """
        # Получаем эмбеддинги
        emb1 = self._get_text_embedding(text1, strategy=strategy)
        emb2 = self._get_text_embedding(text2, strategy=strategy)
        emb3 = self._get_text_embedding(text3, strategy=strategy)
        
        # Собираем в матрицу
        embeddings = np.vstack([emb1, emb2, emb3])
        
        # Вычисляем косинусную близость (матрица 3x3)
        # Значения в диапазоне [-1, 1]
        similarities = cosine_similarity(embeddings)
        
        # Упаковываем в словарь
        result = {
            ('text1', 'text2'): similarities[0, 1],
            ('text1', 'text3'): similarities[0, 2],
            ('text2', 'text3'): similarities[1, 2],
        }
        
        return result
    
    def _lemmatize(self, word: str) -> str:
        """Приводит слово к начальной форме"""
        parsed = self.morph.parse(word.lower())[0]
        return parsed.normal_form

    def measure_abstractness(self, text: str) -> float:
        """Возвращает среднюю абстрактность текста"""

        print(f"DEBUG: abstractness_dict has {len(self.abstractness_dict)} entries")
        print(f"DEBUG: dict keys sample: {list(self.abstractness_dict.keys())[:5]}")
        
        print(text)

        words = re.findall(r'[а-яёА-ЯЁ]+', text.lower())
        stop_words = set(stopwords.words('ru'))
        
        filtered = []
        for word in words:
            print(word)
            if word not in stop_words:
                # Лемматизируем через pymorphy3
                parsed = self.morph.parse(word)[0]
                lemma = parsed.normal_form
                print(lemma)
                filtered.append(lemma)

        print(filtered)
        
        if not filtered:
            return 0.0
        
        values = []
        for token in filtered:
            if token in self.abstractness_dict:
                values.append(self.abstractness_dict[token])
        
        return sum(values) / len(values) if values else 0.0

    def measure_word_frequency(self, text: str) -> Dict[str, float]:
        """Возвращает среднюю частоту и долю известных слов."""
        stop_words = set(stopwords.words('ru'))
        tokens = tokenize(text, 'ru')
        
        filtered = [word for word in tokens if word not in stop_words and word.isalpha()]
        
        if not filtered:
            return {'mean_frequency': 0.0, 'coverage': 0.0}
        
        print(filtered)
        
        freqs = []
        known_count = 0
        
        for word in filtered:
            freq = zipf_frequency(word, 'ru')
            if freq > 0:
                freqs.append(freq)
                known_count += 1

        print(sum(freqs) / len(freqs) if freqs else 0.0)
        
        coverage = known_count / len(filtered) if filtered else 0.0
        
        return {
            'mean_frequency': sum(freqs) / len(freqs) if freqs else 0.0,
            'coverage': coverage
        }

    @torch.no_grad()
    def measure_sentiment(self, text: str) -> str:
        """
        Анализирует тональность текста с использованием Hugging Face модели.
        
        Args:
            text: Входной текст для анализа
            
        Returns:
            Строка с меткой тональности
        """
        if not self._load_sentiment_model():
            return "NEUTRAL"
        
        # Токенизация текста
        inputs = self.sentiment_tokenizer(
            text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Инференс
        outputs = self.sentiment_model(**inputs)
        
        # Применяем softmax для получения вероятностей
        probs = F.softmax(outputs.logits, dim=1)
        
        # Получаем предсказанный класс
        pred_id = torch.argmax(probs, dim=1).item()
        
        # Возвращаем метку
        if self.sentiment_labels:
            return self.sentiment_labels.get(pred_id, str(pred_id))
        else:
            return str(pred_id)
    
    def measure_coherence(self):
        pass

    def measure_linear_semantic_coherence(self):
        pass

    def measure_syntactic_features(self):
        pass

    def measure_stylometric_features(self):
        pass

    def run_evaluation_pipeline(self, text: str, original: str, literal: str, idioms: List[str], metrics=None):
        """
        Запускает пайплайн оценки для текста с идиомами.
        
        Args:
            text: Преобразованный текст с идиомами
            original: Оригинальная версия текста
            literal: Буквальная версия текста
            idioms: Список выделенных идиом в тексте
            metrics: Список метрик для расчета (опционально)
        
        Returns:
            DataFrame с результатами оценки
        """
        if metrics is None:
            metrics = ['perplexity', 'surprisal', 'abstractness', 'word_frequency', 
                    'sentiment', 'cosine_distances']
        
        results = {}
        
        # 1. Метрики на преобразованном тексте (идиомы заменены)
        for name in self._progress(metrics, "Metrics"):
            if name == 'perplexity':
                results['perplexity_score'] = self.measure_perplexity(text)
            
            elif name == 'surprisal':
                # Для идиом вычисляем сюрпрайз
                if idioms:
                    surprisal_results = self.measure_phrase_surprisal(text, idioms)
                    # Сохраняем средний сюрпрайз по всем идиомам
                    avg_surprisal = np.mean([r[1] for r in surprisal_results if r[1] != float('inf')])
                    results['idiom_surprisal_score'] = avg_surprisal if not np.isnan(avg_surprisal) else float('inf')
                    # Также сохраняем детали
                    results['idiom_surprisal_details'] = surprisal_results
                else:
                    results['idiom_surprisal_score'] = float('inf')
            
            elif name == 'abstractness':
                results['abstractness_score'] = self.measure_abstractness(text)
            
            elif name == 'word_frequency':
                freq_stats = self.measure_word_frequency(text)
                results['word_frequency_mean'] = freq_stats['mean_frequency']
                results['word_frequency_coverage'] = freq_stats['coverage']
            
            elif name == 'sentiment':
                results['sentiment_score'] = self.measure_sentiment(text)
            
            elif name == 'cosine_distances':
                # Попарные косинусные расстояния между тремя версиями
                if original and literal:
                    distances = self.measure_pairwise_cosine_distance(text, original, literal)
                    results['cosine_sim_original_vs_text'] = distances[('text1', 'text2')]
                    results['cosine_sim_original_vs_literal'] = distances[('text1', 'text3')]
                    results['cosine_sim_text_vs_literal'] = distances[('text2', 'text3')]
                else:
                    results['cosine_sim_original_vs_text'] = 0.0
                    results['cosine_sim_original_vs_literal'] = 0.0
                    results['cosine_sim_text_vs_literal'] = 0.0
            
            elif name == 'all_metrics':
                # Комплексная оценка всех метрик для текста
                results['perplexity_score'] = self.measure_perplexity(text)
                
                if idioms:
                    surprisal_results = self.measure_phrase_surprisal(text, idioms)
                    avg_surprisal = np.mean([r[1] for r in surprisal_results if r[1] != float('inf')])
                    results['idiom_surprisal_score'] = avg_surprisal if not np.isnan(avg_surprisal) else float('inf')
                    results['idiom_surprisal_details'] = surprisal_results
                else:
                    results['idiom_surprisal_score'] = float('inf')
                
                results['abstractness_score'] = self.measure_abstractness(text)
                
                freq_stats = self.measure_word_frequency(text)
                results['word_frequency_mean'] = freq_stats['mean_frequency']
                results['word_frequency_coverage'] = freq_stats['coverage']
                
                results['sentiment_score'] = self.measure_sentiment(text)
                
                if original and literal:
                    distances = self.measure_pairwise_cosine_distance(original, text, literal)
                    results['cosine_sim_original_vs_text'] = distances[('text1', 'text2')]
                    results['cosine_sim_original_vs_literal'] = distances[('text1', 'text3')]
                    results['cosine_sim_text_vs_literal'] = distances[('text2', 'text3')]
        
        # Преобразуем в DataFrame
        # Для значений-словарей или списков преобразуем в строки
        df_results = {}
        for key, value in results.items():
            if isinstance(value, (list, dict)) and key not in ['idiom_surprisal_details']:
                # Для простых списков/словарей конвертируем в строку
                df_results[key] = str(value)
            else:
                df_results[key] = value
        
        return pd.DataFrame([df_results])




