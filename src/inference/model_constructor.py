import os
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_gigachat.chat_models import GigaChat

load_dotenv()

class ModelConstructor:
    """Фабрика для создания клиентов языковых моделей различных провайдеров.
    
    Предоставляет статические методы для инициализации моделей от разных
    поставщиков (Ollama, Gigachat, Openrouter) с единым интерфейсом.
    """

    @staticmethod
    def create_client(model_name: str, provider: str, **kwargs):
        """Создаёт клиента для указанного провайдера.
        
        Args:
            model_name: идентификатор модели у провайдера
            provider: название провайдера ('openrouter' или 'ollama')
            **kwargs: дополнительные параметры для инициализации модели
                (температура, max_tokens и т.д.)
        
        Returns:
            инициализированный клиент модели
            
        Raises:
            ValueError: если указан неподдерживаемый провайдер
        """
        if provider == "gigachat":
            return GigaChat(
                credentials=os.getenv("GIGACHAT_API_KEY"),
                verify_ssl_certs=False,
                model=model_name
            )
        
        elif provider == "ollama":
            return ChatOllama(model=model_name, **kwargs)
        
        elif provider == "openrouter":
            return ChatOpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                model=model_name,
                **kwargs
            )
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")