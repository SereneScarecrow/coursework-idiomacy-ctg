"""
Модуль для оценки текстов.
"""

import re
import torch
import pandas as pd
from tqdm import tqdm
from typing import Union, List, Optional
from dataclasses import dataclass

from bert_score import BERTScorer
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import nltk
from gensim.models import KeyedVectors

from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer

from evaluation.model_loader import ModelConfig, ModelLoader

try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    download_model = load_from_checkpoint = None

nltk.download('punkt', quiet=True)

class Evaluator:
    
    def __init__(self, device: Optional[str] = None, verbose: bool = True, max_length: int = 512):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        self.max_length = max_length
    
    def _validate_inputs(self, references, candidates):
        if isinstance(references, pd.Series):
            references = references.tolist()
        if isinstance(candidates, pd.Series):
            candidates = candidates.tolist()
        if len(references) != len(candidates):
            raise ValueError("Length mismatch")
        return references, candidates
    
    def _progress(self, data, desc):
        """Прогресс-бар для итераций."""
        return tqdm(data, desc=desc, disable=not self.verbose)
    
    def _score(self, refs, cands, metric_fn, desc):
        """Обертка для метрик, требующих попарного вычисления с прогресс-баром."""
        return [metric_fn(r, c) for r, c in self._progress(zip(refs, cands), desc)]
    
    def _split_into_chunks(self, text: str, strategy: str = 'auto') -> List[str]:
        """
        Разбивает текст на чанки не длиннее max_length токенов.
        
        Стратегии:
        - 'auto': сначала по абзацам (\n), затем по предложениям (.!?;:)
        - 'paragraphs': только по абзацам
        - 'sentences': только по предложениям
        """
        
        if strategy == 'paragraphs':
            units = [p.strip() for p in text.split('\n') if p.strip()]
        elif strategy == 'sentences':
            units = re.split(r'(?<=[.!?;:])\s+', text)
            units = [u.strip() for u in units if u.strip()]
        else:  # 'auto'
            # Сначала пробуем разбить по абзацам
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            if len(paragraphs) > 1:
                units = paragraphs
            else:
                # Если нет абзацев — по предложениям
                units = re.split(r'(?<=[.!?;:])\s+', text)
                units = [u.strip() for u in units if u.strip()]
        
        # Собираем чанки, не превышающие max_length
        chunks = []
        current_chunk = []
        current_length = 0
        
        for unit in units:
            # Оцениваем длину unit в токенах (приблизительно по словам)
            unit_length = len(unit.split())
            
            if current_length + unit_length <= self.max_length:
                current_chunk.append(unit)
                current_length += unit_length
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [unit]
                current_length = unit_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def run_evaluation_pipeline(self):
        raise NotImplementedError


class QualityEvaluator(Evaluator):
    
    def __init__(self, gpu: bool = True, verbose: bool = True, config: ModelConfig = None):
        super().__init__(verbose=verbose)
        self.device = 'cuda' if (gpu and torch.cuda.is_available()) else 'cpu'
        self.config = config or ModelConfig()
        self.loader = ModelLoader()
        
        # Загрузка через ModelLoader
        self.bertscorer = self.loader.load_bert_scorer(
            lang='ru', model_type=self.config.bertscore, device=self.device
        )
        
        bleurt = self.loader.load_bleurt(self.config.bleurt, device=self.device)
        if bleurt:
            self.bleurt_model = bleurt['model']
            self.bleurt_tokenizer = bleurt['tokenizer']
            self.bleurt_scorer = True
        else:
            self.bleurt_scorer = False
        
        self.comet_model = self.loader.load_comet(self.config.comet, device=self.device)
        
        # Word2Vec грузится лениво
        self.w2v_model = None
    
    def _w2v(self):
        if self.w2v_model is None:
            self.w2v_model = self.loader.load_w2v(self.config.w2v)
    
    def measure_bertscore(self, references, candidates):
        if self.bertscorer is None:
            return [0.0] * len(references)
        try:
            _, _, F1 = self.bertscorer.score(candidates, references)
            return F1
        except Exception as e:
            if self.verbose: print(f"[ERROR] BERTScorer: {e}")
            return [0.0] * len(references)
    
    def measure_bleurt(self, references, candidates):
        if not self.bleurt_scorer:
            return [0.0] * len(references)
        
        scores = []
        for ref, cand in zip(references, candidates):
            try:
                # Разбиваем длинные тексты на чанки
                ref_chunks = self._split_into_chunks(ref)
                cand_chunks = self._split_into_chunks(cand)
                
                # Выравниваем количество чанков
                if len(ref_chunks) != len(cand_chunks):
                    min_len = min(len(ref_chunks), len(cand_chunks))
                    ref_chunks = ref_chunks[:min_len]
                    cand_chunks = cand_chunks[:min_len]
                
                if not ref_chunks:
                    scores.append(0.0)
                    continue
                
                # Оцениваем каждый чанк и усредняем
                chunk_scores = []
                for r, c in zip(ref_chunks, cand_chunks):
                    inputs = self.bleurt_tokenizer(
                        r, c,
                        padding='longest',
                        truncation=True,
                        max_length=512,
                        return_tensors='pt'
                    )
                    if self.device == 'cuda':
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    with torch.no_grad():
                        logits = self.bleurt_model(**inputs).logits
                    chunk_scores.append(logits.flatten().cpu().item())
                
                scores.append(sum(chunk_scores) / len(chunk_scores))
                
            except Exception as e:
                if self.verbose:
                    print(f"[ERROR] BLEURT: {e}")
                scores.append(0.0)
        
        return scores
    
    def measure_comet(self, references, candidates):
        if self.comet_model is None:
            return [0.0] * len(references)
        try:
            data = [{"src": "", "mt": c, "ref": r} for r, c in zip(references, candidates)]
            return self.comet_model.predict(data, batch_size=8, 
                    gpus=1 if self.device == 'cuda' else 0).scores
        except Exception as e:
            if self.verbose: print(f"[ERROR] COMET: {e}")
            return [0.0] * len(references)
    
    def measure_meteor(self, references, candidates):
        return self._score(references, candidates,
                          lambda r, c: meteor_score([word_tokenize(r.lower())], word_tokenize(c.lower())),
                          "METEOR")
    
    def measure_wmd(self, references, candidates):
        self._w2v()
        if self.w2v_model is None:
            return [0.0] * len(references)
        return self._score(references, candidates,
                          lambda r, c: 1 / (1 + self.w2v_model.wmdistance(r.split(), c.split())),
                          "WMD")
    
    def run_evaluation_pipeline(self, references, candidates, metrics=None):
        refs, cands = self._validate_inputs(references, candidates)
        
        if metrics is None:
            metrics = ['bertscore', 'bleurt', 'comet', 'meteor', 'wmd']
        
        metric_map = {
            'bertscore': self.measure_bertscore,
            'bleurt': self.measure_bleurt,
            'comet': self.measure_comet,
            'meteor': self.measure_meteor,
            'wmd': self.measure_wmd
        }
        
        results = {}
        for name in self._progress(metrics, "Metrics"):
            if name in metric_map:
                results[f"{name}_score"] = metric_map[name](refs, cands)
            else:
                results[f"{name}_score"] = [0.0] * len(refs)
        
        return pd.DataFrame(results)


# class IdiomEvaluator(Evaluator):
#     def __init__(self):

#     def measure_perplexity():
#         return

#     def measure_surprisal():
#         return
    
#     def measure_cosine_similarity():

#     def measure_coherence():
#         return
    
#     def measure_abstractness():
#         return

