import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Загрузка модели (один раз)
def load_sentiment_model(model_name='blanchefort/rubert-base-cased-sentiment', device='cpu'):
    """
    Загружает модель для анализа тональности.
    
    Returns:
        tuple: (model, tokenizer, labels)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    if device == 'cuda':
        model = model.cuda()
    
    model.eval()
    
    # Метки тональности
    labels = {0: "neutral", 1: "positive", 2: "negative"}
    
    return model, tokenizer, labels


@torch.no_grad()
def measure_sentiment(text: str, model, tokenizer, device='cpu') -> str:
    """
    Анализирует тональность текста.
    
    Args:
        text: Входной текст
        model: Загруженная модель
        tokenizer: Токенизатор
        device: 'cpu' или 'cuda'
    
    Returns:
        str: 'POSITIVE', 'NEUTRAL', или 'NEGATIVE'
    """
    inputs = tokenizer(
        text,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors='pt'
    ).to(device)
    
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    pred_id = torch.argmax(probs, dim=1).item()
    
    labels = {0: "neutral", 1: "positive", 2: "negative"}
    return labels[pred_id]

# Использование с DataFrame
if __name__ == "__main__":
    import pandas as pd
    
    # Загружаем модель один раз
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer, _ = load_sentiment_model(device=device)
    
    # Читаем данные
    df = pd.read_csv("data/all_generated_metrics.csv")
    
    # Применяем функцию
    df['sentiment'] = df['text'].apply(lambda x: measure_sentiment(x, model, tokenizer, device))
    
    print(df[['text', 'sentiment']])

    df.to_csv("data/all_generated_metrics.csv")