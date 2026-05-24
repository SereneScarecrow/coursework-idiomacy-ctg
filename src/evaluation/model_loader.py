"""
Модуль для загрузки моделей.
"""

import torch
from dataclasses import dataclass
from typing import Optional

from bert_score import BERTScorer
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
from gensim.models import KeyedVectors

try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    download_model = load_from_checkpoint = None


@dataclass
class ModelConfig:
    bertscore: str = 'bert-base-multilingual-cased'
    bleurt: str = 'lucadiliello/BLEURT-20-D12'
    comet: str = 'Unbabel/wmt22-comet-da'
    w2v: Optional[str] = None


class ModelLoader:
    """Унифицированная загрузка моделей."""
    
    @staticmethod
    def _warn(msg):
        print(f"Warning: {msg}")
    
    @staticmethod
    def load_bert_scorer(lang='ru', model_type=None, device='cpu'):
        try:
            return BERTScorer(lang=lang, model_type=model_type, 
                              num_layers=8, device=device)
        except Exception as e:
            ModelLoader._warn(f"BERTScorer failed: {e}")
            return None
    
    @staticmethod
    def load_bleurt(model_path, device='cpu'):
        try:
            config = BleurtConfig.from_pretrained(model_path)
            model = BleurtForSequenceClassification.from_pretrained(model_path)
            tokenizer = BleurtTokenizer.from_pretrained(model_path)
            if device == 'cuda':
                model = model.cuda()
            model.eval()
            return {'model': model, 'tokenizer': tokenizer}
        except Exception as e:
            ModelLoader._warn(f"BLEURT failed: {e}")
            return None
    
    @staticmethod
    def load_comet(model_path, device='cpu'):
        if not COMET_AVAILABLE:
            ModelLoader._warn("COMET package not installed")
            return None
        try:
            path = download_model(model_path, saving_directory='./comet_models')
            model = load_from_checkpoint(path)
            if device == 'cuda':
                model = model.cuda()
            return model
        except Exception as e:
            ModelLoader._warn(f"COMET failed: {e}")
            return None
    
    @staticmethod
    def load_w2v(model_path=None):
        try:
            if model_path:
                return KeyedVectors.load_word2vec_format(model_path, limit=200000)
            else:
                from gensim.downloader import load
                return load('glove-twitter-25')
        except Exception as e:
            ModelLoader._warn(f"Word2Vec failed: {e}")
            return None