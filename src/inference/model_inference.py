import os
from dotenv import load_dotenv

# LangChain и LangFuse импорты
from langfuse import Langfuse
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler

load_dotenv()


class ModelConstructor:
    """Фабрика для создания клиентов языковых моделей различных провайдеров.
    
    Предоставляет статические методы для инициализации моделей от разных
    поставщиков (OpenRouter, Ollama) с единым интерфейсом.
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
            ChatOpenAI или ChatOllama: инициализированный клиент модели
            
        Raises:
            ValueError: если указан неподдерживаемый провайдер
        """
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
    """Конструктор промптов из LangFuse.
    
    Извлекает промпты из LangFuse по имени и версии, компилирует их
    с переданными переменными в готовые шаблоны LangChain.
    """

    @staticmethod
    def get_prompt(prompt_name: str, variables: dict, version=None):
        """Получает промпт из LangFuse и возвращает LangChain шаблон.
        
        Args:
            prompt_name: имя промпта в LangFuse
            variables: словарь переменных для подстановки в промпт
            version: версия промпта (если не указана, берётся последняя)
        
        Returns:
            LangChain шаблон промпта с подставленными переменными
            
        Note:
            Промпт должен быть совместим с методом compile() LangFuse
        """
        langfuse = Langfuse()

        if version:
            prompt = langfuse.get_prompt(prompt_name, version=version)
        else:
            prompt = langfuse.get_prompt(prompt_name)

        if hasattr(prompt, 'compile'):
            return prompt.compile(**variables)


class ModelInference:
    """Инференс языковой модели с поддержкой логирования в LangFuse.
    
    Оборачивает модель и предоставляет единый интерфейс для вызова
    с автоматической обработкой callback'ов LangFuse.
    
    Attributes:
        model: инициализированная модель для инференса
        langfuse_handler: обработчик callback'ов для LangFuse
    """

    def __init__(self, model: ModelConstructor,
                 langfuse_handler: CallbackHandler | None = None):
        """Инициализирует инференс моделью и опциональным LangFuse handler'ом.
        
        Args:
            model: экземпляр модели, созданный через ModelConstructor
            langfuse_handler: обработчик для логирования в LangFuse
        """
        self.model = model
        self.langfuse_handler = langfuse_handler

    def __call__(self, prompt_text: str, **kwargs):
        """Запускает инференс с переданным промптом.
        
        Args:
            prompt_text: готовый текст промпта (уже с подставленными переменными)
            **kwargs: дополнительные параметры для invoke (например, config)
        
        Returns:
            str: текст ответа модели
            
        Note:
            Если ответ модели имеет атрибут 'content' (стандарт LangChain),
            возвращается он, иначе ответ преобразуется в строку.
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