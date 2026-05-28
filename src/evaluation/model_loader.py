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
    # качество генерации
    bertscore: str = 'bert-base-multilingual-cased'
    bleurt: str = 'lucadiliello/BLEURT-20-D12'
    comet: str = 'Unbabel/wmt22-comet-da'
    w2v: Optional[str] = None
    # идиоматичность
    perplexity_model: str = 'ai-forever/rugpt3medium_based_on_gpt2'
    embedding_model: str = 'intfloat/multilingual-e5-large'  # или 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    embedding_strategy: str = 'mean'  # 'mean' или 'cls'
    sentiment_model = 'blanchefort/rubert-base-cased-sentiment'
    sentiment_labels: dict = None


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
        
    @staticmethod
    def load_perplexity_model(model_name: str = 'cointegrated/ruGPT-3.5-13B', device: str = 'cpu'):
        """
        Загружает языковую модель для расчета перплексии.
        
        Args:
            model_name: Имя модели на Hugging Face
            device: 'cpu' или 'cuda'
        
        Returns:
            tuple: (model, tokenizer) или (None, None) при ошибке
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
                low_cpu_mem_usage=True
            )
            
            if device == 'cuda':
                model = model.cuda()
            
            model.eval()
            
            # Устанавливаем pad_token, если его нет
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            return model, tokenizer
            
        except Exception as e:
            ModelLoader._warn(f"Failed to load perplexity model {model_name}: {e}")
            return None, None
        
    @staticmethod
    def load_embedding_model(model_name: str = 'intfloat/multilingual-e5-large', device: str = 'cpu'):
        """
        Загружает модель для получения эмбеддингов текстов.
        
        Args:
            model_name: Имя модели на Hugging Face
            device: 'cpu' или 'cuda'
        
        Returns:
            tuple: (model, tokenizer) или (None, None) при ошибке
        """
        try:
            from transformers import AutoModel, AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
            if device == 'cuda':
                model = model.cuda()
            
            model.eval()
            
            return model, tokenizer
            
        except Exception as e:
            ModelLoader._warn(f"Failed to load embedding model {model_name}: {e}")
            return None, None
    
    @staticmethod
    def load_sentiment_model(model_name: str = 'blanchefort/rubert-base-cased-sentiment', 
                            device: str = 'cpu'):
        """
        Загружает модель для анализа тональности.
        
        Args:
            model_name: Имя модели на Hugging Face
            device: 'cpu' или 'cuda'
        
        Returns:
            tuple: (model, tokenizer, labels) или (None, None, None) при ошибке
        """
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            if device == 'cuda':
                model = model.cuda()
            
            model.eval()
            
            # Определяем метки классов
            if hasattr(model.config, 'id2label') and model.config.id2label:
                labels = model.config.id2label
            else:
                # Стандартные метки для моделей тональности
                labels = {0: "neutral", 1: "positive", 2: "negative"}
            
            return model, tokenizer, labels
            
        except Exception as e:
            ModelLoader._warn(f"Failed to load sentiment model {model_name}: {e}")
            return None, None, None