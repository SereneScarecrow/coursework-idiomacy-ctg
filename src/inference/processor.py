import os
import pandas as pd
from tqdm import tqdm
from typing import Optional, Callable, Dict, Any, Union

from langfuse.langchain import CallbackHandler
from inference.model_constructor import ModelConstructor
from inference.model_inference import PromptConstructor, ModelInference


class CheckpointProcessor:
    """
    Универсальный класс для обработки текстов с поддержкой чекпоинтов,
    структурированного вывода и постобработки.
    """
    
    def __init__(
        self,
        model_name: str,
        provider: str,
        prompt_name: str,
        prompt_version: Optional[int] = None,
        structured_output_schema: Optional[Any] = None,
        checkpoint_path: str = "checkpoint.csv",
        result_column_name: str = "result",
        **model_kwargs
    ):
        """
        Args:
            model_name: имя модели
            provider: провайдер
            prompt_name: имя промпта в LangFuse
            prompt_version: версия промпта (если не указана, берётся последняя)
            structured_output_schema: Pydantic схема для структурированного вывода
            checkpoint_path: путь к чекпоинту
            result_column_name: имя колонки для результатов
            **model_kwargs: доп. аргументы для модели (temperature, top_p и т.д.)
        """
        self.model_name = model_name
        self.provider = provider
        self.prompt_name = prompt_name
        self.prompt_version = prompt_version
        self.structured_output_schema = structured_output_schema
        self.checkpoint_path = checkpoint_path
        self.result_column_name = result_column_name
        self.model_kwargs = model_kwargs
        
        # инициализация клиента и инференса
        self._init_model()
    
    def _init_model(self):
        """Инициализирует модель и инференс"""
        langfuse_handler = CallbackHandler()
        
        client = ModelConstructor.create_client(
            self.model_name,
            self.provider,
            structured_output_schema=self.structured_output_schema,
            **self.model_kwargs
        )
        
        self.inference = ModelInference(client, langfuse_handler=langfuse_handler)
        self.prompt_builder = PromptConstructor(self.prompt_name, version=self.prompt_version)
    
    def process(
        self,
        df: pd.DataFrame,
        text_column: str = 'text',
        columns_mapping: Optional[Dict[str, str]] = None,
        post_process_fn: Optional[Callable] = None,
        resume: bool = True
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Обрабатывает датафрейм с чекпоинтами.
        
        Args:
            df: датафрейм с данными
            text_column: имя колонки при передаче Series
            columns_mapping: переименование колонок DataFrame перед рендерингом
            post_process_fn: функция постобработки результата
            resume: возобновлять ли с чекпоинта
        
        Returns:
            Series (для текста) или DataFrame (для структурированного вывода)
        """
        # рендер промптов с поддержкой mapping
        texts_to_process = self.prompt_builder.render_from_df(
            df, 
            text_column=text_column,
            columns_mapping=columns_mapping
        )
        
        # загрузка из чекпоинта - теперь с сохранением порядка
        results, processed_indices = self._load_checkpoint(resume, len(texts_to_process), post_process_fn)
        
        # обработка
        for idx, text in tqdm(enumerate(texts_to_process), total=len(texts_to_process)):
            if idx in processed_indices:
                continue
            
            result = self._process_single(text, post_process_fn, idx)
            results[idx] = result  # сохраняем по правильному индексу
            self._save_checkpoint(results)
        
        # форматирование результата
        return self._format_results(results)
    
    def process_single(self, variables: Dict[str, Any], post_process_fn: Optional[Callable] = None) -> Any:
        """
        Обрабатывает один промпт без чекпоинта.
        
        Args:
            variables: словарь переменных для подстановки в промпт
            post_process_fn: функция постобработки результата
        
        Returns:
            Обработанный результат
        """
        text = self.prompt_builder.render(variables)
        return self._process_single(text, post_process_fn, idx=-1)
    
    def _process_single(self, text: str, post_process_fn: Optional[Callable], idx: int) -> Any:
        """Обрабатывает один текст с обработкой ошибок"""
        try:
            raw_result = self.inference(text)
            
            if post_process_fn:
                return post_process_fn(raw_result)
            return raw_result
        except Exception as e:
            print(f"Ошибка на индексе {idx}: {e}")
            return None
    
    def _load_checkpoint(self, resume: bool, total_size: int, post_process_fn: Optional[Callable]) -> tuple:
        """
        Загружает чекпоинт и возвращает (results, processed_indices)
        Теперь сохраняет порядок с помощью списка фиксированного размера
        """
        # Инициализируем results списком нужного размера с None
        results = [None] * total_size
        processed_indices = set()
        
        if not resume or not os.path.exists(self.checkpoint_path):
            return results, processed_indices
        
        saved_df = pd.read_csv(self.checkpoint_path)
        if saved_df.empty or self.result_column_name not in saved_df.columns:
            return results, processed_indices
        
        # Проверяем, есть ли колонка 'original_index' для восстановления порядка
        if 'original_index' in saved_df.columns:
            # Восстанавливаем порядок по original_index
            for _, row in saved_df.iterrows():
                original_idx = int(row['original_index'])
                if original_idx < total_size:  # проверка на валидность
                    val = row[self.result_column_name]
                    if pd.notna(val) and val is not None:
                        # применяем постобработку к загруженным данным если нужно
                        if post_process_fn:
                            try:
                                val = post_process_fn(val)
                            except:
                                pass
                        results[original_idx] = val
                        processed_indices.add(original_idx)
        else:
            # Старый формат - пытаемся восстановить порядок как можем
            for i, row in saved_df.iterrows():
                if i >= total_size:
                    break
                val = row[self.result_column_name]
                if pd.notna(val) and val is not None:
                    if post_process_fn:
                        try:
                            val = post_process_fn(val)
                        except:
                            pass
                    results[i] = val
                    processed_indices.add(i)
        
        processed_count = len([r for r in results if r is not None])
        print(f"Загружено {processed_count} результатов из {len(saved_df)} сохраненных")
        return results, processed_indices
    
    def _save_checkpoint(self, results: list):
        """Сохраняет чекпоинт с сохранением исходных индексов"""
        # Создаем список только с не-None результатами для эффективности
        # но сохраняем original_index для восстановления порядка
        checkpoint_data = {
            'original_index': [],  # сохраняем исходный индекс
            self.result_column_name: []
        }
        
        for idx, result in enumerate(results):
            if result is not None:
                checkpoint_data['original_index'].append(idx)
                checkpoint_data[self.result_column_name].append(result)
        
        if checkpoint_data['original_index']:  # только если есть данные
            pd.DataFrame(checkpoint_data).to_csv(self.checkpoint_path, index=False)
    
    def _format_results(self, results: list) -> Union[pd.Series, pd.DataFrame]:
        """Форматирует результаты в Series или DataFrame, сохраняя порядок"""
        # Фильтруем None значения на случай, если что-то пошло не так
        # но сохраняем исходную позицию
        if not self.structured_output_schema:
            # Для Series возвращаем с исходными индексами
            series_data = []
            for res in results:
                series_data.append(res)
            return pd.Series(series_data, name=self.result_column_name)
        
        # для структурированного вывода - сохраняем порядок строк
        rows = []
        for i, res in enumerate(results):
            if res is None:
                rows.append({'index': i})  # помечаем как None, но сохраняем позицию
            elif hasattr(res, 'model_dump'):
                row_data = res.model_dump()
                row_data['index'] = i
                rows.append(row_data)
            elif hasattr(res, 'dict'):
                row_data = res.dict()
                row_data['index'] = i
                rows.append(row_data)
            elif isinstance(res, dict):
                row_data = res.copy()
                row_data['index'] = i
                rows.append(row_data)
            else:
                rows.append({'index': i, self.result_column_name: res})
        
        df = pd.DataFrame(rows)
        return df