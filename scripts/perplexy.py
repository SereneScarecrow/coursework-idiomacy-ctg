# perplexity_surprisal.py

import torch
import re
from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==================== ЗАГРУЗКА МОДЕЛИ ====================

def load_perplexity_model(model_name: str = 'sberbank-ai/rugpt3medium_based_on_gpt2', device: str = 'cpu'):
    """
    Загружает языковую модель для расчета перплексии и сюрпрайза.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if device == 'cuda':
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16
            )
            model = model.cuda()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32
            )
            model = model.to(device)
        
        # Устанавливаем pad_token если его нет
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.eval()
        print(f"Model loaded successfully on {device}")
        return model, tokenizer
        
    except Exception as e:
        print(f"Failed to load perplexity model: {e}")
        return None, None


# ==================== ПЕРПЛЕКСИЯ ====================

@torch.no_grad()
def measure_perplexity(text: str, model, tokenizer, device: str = 'cpu') -> float:
    """Вычисляет перплексию текста."""
    if model is None or tokenizer is None:
        return float('inf')
    
    encodings = tokenizer(text, return_tensors='pt', add_special_tokens=True)
    input_ids = encodings['input_ids'].to(device)
    
    outputs = model(input_ids, labels=input_ids)
    
    return torch.exp(outputs.loss).item()


# ==================== СЮРПРАЙЗ ДЛЯ ФРАЗ ====================

def find_phrase_spans(text: str, target_phrases: List[str]) -> List[Tuple[str, int, int]]:
    """Находит все вхождения целевых фраз в тексте."""
    spans = []
    for phrase in target_phrases:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        for match in pattern.finditer(text):
            spans.append((match.group(), match.start(), match.end()))
    return spans


def align_spans_with_tokens(text: str, spans: List[Tuple[str, int, int]], tokenizer) -> List[dict]:
    """Выравнивает spans с токенами."""
    encodings = tokenizer(
        text,
        return_tensors='pt',
        return_offsets_mapping=True,
        add_special_tokens=True
    )
    
    offsets = encodings['offset_mapping'][0]
    input_ids = encodings['input_ids'][0]
    
    result = []
    for phrase, start_char, end_char in spans:
        token_indices = []
        tokens = []
        
        for i, (start, end) in enumerate(offsets):
            if start < end_char and end > start_char:
                token_indices.append(i)
                token = tokenizer.decode([input_ids[i]])
                tokens.append(token)
        
        result.append({
            'text': phrase,
            'token_indices': token_indices,
            'tokens': tokens
        })
    
    return result


@torch.no_grad()
def compute_token_surprisal_sequence(text: str, token_indices: List[int], model, tokenizer, device: str = 'cpu') -> List[float]:
    """Вычисляет сюрпрайз для последовательности токенов."""
    if not token_indices or model is None:
        return [float('inf')] * len(token_indices)
    
    encodings = tokenizer(
        text,
        return_tensors='pt',
        add_special_tokens=True,
        padding=False,
        truncation=True
    )
    
    input_ids = encodings['input_ids'].to(device)
    
    outputs = model(input_ids)
    logits = outputs.logits
    
    surprisals = []
    for token_pos in token_indices:
        if token_pos == 0:
            continue
        
        logits_for_pred = logits[0, token_pos - 1, :]
        log_probs = torch.log_softmax(logits_for_pred, dim=-1)
        true_token_id = input_ids[0, token_pos]
        
        surprisal = -log_probs[true_token_id].item()
        surprisals.append(surprisal)
    
    return surprisals


def measure_phrase_surprisal(text: str, target_phrases: List[str], model, tokenizer, device: str = 'cpu') -> List[Tuple[str, float, List[float]]]:
    """Вычисляет сюрпрайз для выделенных фраз."""
    if not target_phrases:
        return []
    
    spans = find_phrase_spans(text, target_phrases)
    
    if not spans:
        return [(phrase, float('inf'), []) for phrase in target_phrases]
    
    aligned = align_spans_with_tokens(text, spans, tokenizer)
    
    results = []
    for span in aligned:
        token_indices = span['token_indices']
        
        if not token_indices:
            results.append((span['text'], float('inf'), []))
            continue
        
        token_surprisals = compute_token_surprisal_sequence(text, token_indices, model, tokenizer, device)
        
        if token_surprisals:
            avg_surprisal = sum(token_surprisals) / len(token_surprisals)
        else:
            avg_surprisal = float('inf')
        
        results.append((span['text'], avg_surprisal, token_surprisals))
    
    return results


# ==================== ПРИМЕР ====================

if __name__ == "__main__":
    import pandas as pd
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model, tokenizer = load_perplexity_model(
        model_name="Qwen/Qwen2-1.5B",
        device=device
    )
    
    if model is not None:
        # Тестируем
        text = "Это тестовый текст для проверки перплексии"
        ppl = measure_perplexity(text, model, tokenizer, device)
        print(f"Perplexity: {ppl:.2f}")