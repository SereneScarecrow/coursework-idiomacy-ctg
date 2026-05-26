import pandas as pd
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from inference.model_constructor import ModelConstructor

class PromptConstructor:
    """Конструктор промптов из LangFuse.
    
    Загружает промпт из LangFuse по имени и версии при инициализации,
    затем позволяет многократно рендерить его с разными переменными.
    """
    def __init__(self, prompt_name: str, version=None):
        """Инициализирует конструктор промпта.
        
        Args:
            prompt_name: имя промпта в LangFuse
            version: версия промпта (если не указана, берётся последняя)
        """
        self.prompt_name = prompt_name
        self.version = version
        self._raw_prompt = None
        self._load_prompt()

    def _load_prompt(self):
        """Загружает промпт из LangFuse."""
        langfuse = Langfuse()
        
        if self.version:
            self._raw_prompt = langfuse.get_prompt(self.prompt_name, version=self.version)
        else:
            self._raw_prompt = langfuse.get_prompt(self.prompt_name)

    def render(self, variables):
        """Рендерит промпт с переданными переменными.
        
        Args:
            variables: словарь переменных для подстановки в промпт
        
        Returns:
            Скомпилированный промпт с подставленными переменными
            
        Note:
            Промпт должен быть совместим с методом compile() LangFuse
        """
        if hasattr(self._raw_prompt, 'compile'):
            return self._raw_prompt.compile(**variables)
        return self._raw_prompt
    
    def render_from_df(self, df, text_column='text', columns_mapping=None):
        """Рендерит промпт с переменными для каждой строки датафрейма Pandas.
        
        Args:
            df: Pandas DataFrame или Series. 
                - Если DataFrame: названия колонок используются как ключи переменных
                - Если Series: преобразуется в словари с ключом, указанным в text_column
            text_column (str): Имя колонки/ключа при передаче Series. По умолчанию 'text'
        columns_mapping (dict, optional): Словарь для переименования колонок DataFrame
            перед преобразованием. Например: {'original_text': 'text', 'id': 'identifier'}
            По умолчанию None (используются исходные названия колонок)
    
        Returns:
            Список скомпилированных промптов
        """
        if isinstance(df, pd.Series):
            var_dicts = [{text_column: value} for value in df]
        elif isinstance(df, pd.DataFrame):
            # Применяем переименование колонок, если оно задано
            if columns_mapping:
                df = df.rename(columns=columns_mapping)
            var_dicts = df.to_dict('records')
        else:
            raise TypeError(f"Expected pandas DataFrame or Series, got {type(df)}")
        
        return [self.render(var_dict) for var_dict in var_dicts]


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