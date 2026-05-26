import os
from dotenv import load_dotenv

from typing import Type, Optional, List
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_gigachat.chat_models import GigaChat

from pydantic import BaseModel, Field

load_dotenv()


class FormatIdiomatic(BaseModel):
    text: str = Field(description="Сгенерированный текст на русском языке.")
    ratio: float = Field(description="Процент идиоматичности. 0-100, где 0 — нет ни одной идиомы, 100 — все слова текста входят в идиоматичные выражения.", ge=0, le=100)


class FormatJudge(BaseModel):
    grammar: int = Field(
        description=("Грамматическая корректность текста. "
        "Оценка от 0 до 5, где 0 — множество грубых грамматических ошибок, 5 — безупречная грамматика, нет ни одной ошибки."),
        ge=0, le=5
    )
    fluency: int = Field(
        description=("Плавность и естественность речи. "
        "Оценка от 0 до 5, где 0 — неестественные конструкции, рваный ритм, 5 — текст читается легко и плавно, как у носителя."),
        ge=0, le=5
    )
    naturalness: int = Field(
        description=("Естественность и осмысленность употребления идиоматических выражений. "
        "Оценка от 0 до 5, где 0 — идиомы вставлены неестественно, как чужеродные элементы, 5 — идиомы органично вплетены в контекст."),
        ge=0, le=5
    )
    idiomaticity: int = Field(
        description=("Общая образность и насыщенность текста идиомами. "
        "Оценка от 0 до 2, где 0 — текст абсолютно буквальный, ни одной идиомы, 1 — есть несколько идиом, текст образный, 2 — предельно насыщенный идиоматичный текст, где почти каждое предложение содержит образные выражения."),
        ge=0, le=2
    )
    idiomatic_expressions: str = Field(
        description="Полный список всех идиоматических выражений, встреченных в тексте, в той форме, в которой они встретились."
    )


class ModelConstructor:
    """Фабрика для создания клиентов языковых моделей различных провайдеров.
    
    Предоставляет статические методы для инициализации моделей от разных
    поставщиков (Ollama, Gigachat, Openrouter) с единым интерфейсом.
    """

    @staticmethod
    def create_client(model_name: str, provider: str, structured_output_schema: Optional[Type[BaseModel]] = None, **kwargs):
        """Создаёт клиента для указанного провайдера.
        
        Args:
            model_name: идентификатор модели у провайдера
            provider: название провайдера ('openrouter' или 'ollama')
            structured_output_schema: если указан, возвращает структурированного клиента
            с заданной Pydantic-схемой, иначе - обычного клиента
            **kwargs: дополнительные параметры для инициализации модели
                (температура, max_tokens и т.д.)
        
        Returns:
            инициализированный клиент модели
            
        Raises:
            ValueError: если указан неподдерживаемый провайдер
        """
        if provider == "gigachat":
            client = GigaChat(
                credentials=os.getenv("GIGACHAT_API_KEY"),
                verify_ssl_certs=False,
                model=model_name
            )
        
        elif provider == "ollama":
            client = ChatOllama(model=model_name, **kwargs)
        
        elif provider == "openrouter":
            client = ChatOpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                model=model_name,
                **kwargs
            )
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        if structured_output_schema is not None:
            return client.with_structured_output(structured_output_schema, method="json_schema")
