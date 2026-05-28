import re
import torch
import pandas as pd
from typing import List, Tuple, Optional
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import nltk

from evaluation.model_loader import ModelConfig, ModelLoader
from evaluation.evaluator import Evaluator

nltk.download('punkt', quiet=True)

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
    
    def _align_chunks_sequential_all(self, ref_chunks: List[str], cand_chunks: List[str]) -> List[Tuple[str, str]]:
            """Последовательное сопоставление - все чанки, повторяем последний"""
            aligned_pairs = []
            max_len = max(len(ref_chunks), len(cand_chunks))
            
            for i in range(max_len):
                ref = ref_chunks[i] if i < len(ref_chunks) else ref_chunks[-1]
                cand = cand_chunks[i] if i < len(cand_chunks) else cand_chunks[-1]
                aligned_pairs.append((ref, cand))
            
            return aligned_pairs
    
    def measure_bleurt(self, references, candidates):
        if not self.bleurt_scorer:
            return [0.0] * len(references)
        
        scores = []
        for ref, cand in zip(references, candidates):
            try:
                # Используем существующий метод чанкинга из родительского класса
                ref_chunks = self._split_into_chunks(ref, strategy='auto')
                cand_chunks = self._split_into_chunks(cand, strategy='auto')
                
                if not ref_chunks or not cand_chunks:
                    scores.append(0.0)
                    continue
                
                # ВЫБЕРИТЕ ОДИН ИЗ МЕТОДОВ (все используют ВСЕ чанки):
                
                # Вариант 1: последовательно, повторяем последний чанк для недостающих
                aligned_pairs = self._align_chunks_sequential_all(ref_chunks, cand_chunks)
                
                if not aligned_pairs:
                    scores.append(0.0)
                    continue
                
                # Оцениваем каждую выровненную пару
                chunk_scores = []
                for r, c in aligned_pairs:
                    # Пропускаем пустые строки (только для варианта 3)
                    if not r or not c:
                        chunk_scores.append(0.0)
                        continue
                        
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
                
                scores.append(sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0.0)
                
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

    def clean(self, text):
        text = re.sub(r'[^\w\s]', '', text.lower())  # убираем пунктуацию, нижний регистр
        words = [w for w in text.split()]
        return words if words else ['__empty__']  # костыль для пустых текстов

    def measure_wmd(self, references, candidates):
        self._w2v()
        if self.w2v_model is None:
            return [0.0] * len(references)
        return self._score(references, candidates,
                        lambda r, c: 1 / (1 + self.w2v_model.wmdistance(self.clean(r), self.clean(c))),
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