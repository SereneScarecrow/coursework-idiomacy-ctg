import os
import dotenv
from langchain_ollama import ChatOllama
from langchain_openrouter import ChatOpenRouterv

load_dotenv()

class ModelConstructor:
    @staticmethod
    def create_client(model_name: str, provider: str, **kwargs):
        """Создаёт клиента для указанного провайдера"""
        
        if provider == "openrouter":
            return ChatOpenRouterv(model=model_name, **kwargs)
            
        elif provider == "ollama":
            return ChatOllama(model=model_name, **kwargs)
            
        else:
            raise ValueError(f"Unsupported provider: {provider}")