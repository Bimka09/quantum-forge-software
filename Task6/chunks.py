import os
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

MISTRAL_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
try:
    tokenizer = AutoTokenizer.from_pretrained(MISTRAL_MODEL_NAME)
    print(f"Токенизатор '{MISTRAL_MODEL_NAME}' успешно загружен.")
except Exception as e:
    print(
        f"Внимание: tiktoken не может загрузить '{tokenizer_name}'. Установите tiktoken или измените модель. Ошибка: {e}")
    tokenizer = None


def count_tokens(text: str) -> int:
    """Считает количество токенов в тексте с помощью tiktoken"""
    if tokenizer is None:
        # Грубая оценка: ~4 слова ≈ 1 токен (для английского),
        # для русского — лучше использовать 3-4
        return len(text.split()) // 4
    return len(tokenizer.encode(text))


# Создание сплиттера, ориентированного на токены
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,  # Целевой размер чанка — до 800 токенов (в пределах 500–1000)
    chunk_overlap=100,  # Перекрытие для сохранения контекста
    length_function=lambda s: count_tokens(s),  # Используем подсчёт токенов вместо символов
    separators=[
        "\n\n", "\n", ".", "!", "?", " ", ""
    ]  # Разделители по умолчанию, с приоритетом на логические разрывы
)


def split_text_from_file(file_path, source_id=None):
    """Читает текст из файла и возвращает список чанков с метаданными"""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Получаем время последнего изменения файла
    file_mtime = os.path.getmtime(file_path)  # в формате timestamp (float)

    # Определяем источник
    source = file_path if source_id is None else source_id
    # Получаем чанки
    chunks = text_splitter.split_text(text)
    # Добавляем метаданные: откуда взялся чанк и его порядковый номер
    chunks_with_metadata = []
    for i, chunk in enumerate(chunks):
        chunk_data = {
            "text": chunk,
            "source": source,
            "last_updated": file_mtime,
            "chunk_index": i,
            "token_count": count_tokens(chunk),
        }
        chunks_with_metadata.append(chunk_data)
    return chunks_with_metadata


def process_directory(directory_path, file_predicate):
    """Обрабатывает все файлы, удовлетворяющие предикату, в директории"""
    all_chunks = []
    processed_files = []
    for filename in os.listdir(directory_path):
        if file_predicate(filename):
            processed_files.append(filename)
            file_path = os.path.join(directory_path, filename)
            print(f"Processing {filename}...")
            chunks = split_text_from_file(file_path, source_id=filename)
            all_chunks.extend(chunks)
    return all_chunks, processed_files