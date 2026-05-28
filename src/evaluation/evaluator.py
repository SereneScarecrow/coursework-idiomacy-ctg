"""
Модуль для оценки текстов.
"""
import re
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Optional
import nltk

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