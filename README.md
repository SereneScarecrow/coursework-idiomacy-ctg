# Курсовая работа по теме "контролируемая генерация текста с учетом уровня идиоматичности"
В работе описывается разработка и тестирование бенчмарка для задачи figurative language generation.

## Структура проекта
```
coursework-idiomacy-ctg
├── README.md
├── data
│   ├── dataset_maxi_literal.csv # основной датасет бенчмарка
│   └── intermediate # данные процесса создания датасета
├── notebooks
│   ├── dataset_stats.ipynb # статистики dataset_maxi
│   └── eval_prototype.ipynb # прототип кода для оценки бенчмарка
├── prompts # архив промптов
├── requirements.txt # зависимости
├── scripts # вспомогательный код
├── src
│   ├── evaluation # модуль для оценки
│   ├── extract_literal_versions.py # код для извлечения буквальнх версий текстов
│   ├── generate_idiomatic_versions.py # код для генерации образных текстов
│   ├── inference # модуль работы с моделью
│   │   ├── model_constructor.py # инициализатор клиента
│   │   └── model_inference.py # компилятор промпта, запросы к модели
```
