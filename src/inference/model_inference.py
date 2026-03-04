import os
from dotenv import load_dotenv

# LangChain и LangFuse импорты
from langfuse import Langfuse
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler

load_dotenv()

class ModelConstructor:
    @staticmethod
    def create_client(model_name: str, provider: str, **kwargs):
        """Создаёт клиента для указанного провайдера"""
        
        if provider == "openrouter":
            return ChatOpenAI(
                            api_key=os.getenv("OPENROUTER_API_KEY"),
                            base_url="https://openrouter.ai/api/v1",
                            model=model_name,
                            **kwargs
                            )
            
        elif provider == "ollama":
            return ChatOllama(model=model_name, **kwargs)
            
        else:
            raise ValueError(f"Unsupported provider: {provider}")


class PromptConstructor:
    @staticmethod
    def get_prompt(prompt_name: str, variables: dict, version = None):
            """Получает промпт из LangFuse и возвращает LangChain шаблон"""
            
            langfuse = Langfuse()
            
            if version:
                prompt = langfuse.get_prompt(prompt_name, version=version)
            else:
                prompt = langfuse.get_prompt(prompt_name)

            if hasattr(prompt, 'compile'):
                return prompt.compile(**variables)


class ModelInference:
    def __init__(self, model, langfuse_handler=None):
        self.model = model
        self.langfuse_handler = langfuse_handler

    def __call__(self, prompt_text: str, **kwargs):
        """
        Запускает инференс с переданным промптом.
        
        Args:
            prompt_text: готовый текст промпта (уже с подставленными переменными)
            **kwargs: дополнительные параметры для invoke (например, config)
        """
        # Подготавливаем конфиг для callback'ов
        config = {}
        if self.langfuse_handler:
            config = {"callbacks": [self.langfuse_handler]}
        
        # Вызываем модель напрямую
        response = self.model.invoke(prompt_text, config=config, **kwargs)
        
        # Извлекаем текст ответа
        if hasattr(response, 'content'):
            return response.content
        return str(response)