import os
from datetime import datetime

from build_index import get_file_index, generate_and_store_embeddings
from chunks import process_directory

source_dir = "Task6/docs"
logs_dir = "Task6/logs"


# читать файлы
# вычислить обновленные файлы
# разбить файлы на чанки
# вычислить эмбеддинги
# обновить индекс
def update_index():
    operation_context = Context()
    try:
        # получить индекс файлов: какие файлы и когда были обновлены
        file_index = get_file_index()
        # читать обновленные файлы и разбить на чанки
        chunks, processed_files = process_directory(source_dir, lambda x: must_update(x, file_index))
        operation_context.chunks = chunks
        operation_context.processed_files = processed_files
        if len(chunks) != 0:
            # вычислить эмбеддинги и обновить индекс
            generate_and_store_embeddings(chunks)
    except Exception as exception:
        operation_context.error = exception
    operation_context.end_time = datetime.now().strftime("%H:%M:%S")
    return operation_context


# Условие обновления файла: ранее не индексирован или изменен
def must_update(file_path, file_index):
    if file_path not in file_index:
        return file_path.endswith(".txt")
    else:
        current_mtime = os.path.getmtime(source_dir + f"/{file_path}")
        indexed_mtime = file_index[file_path]
        return current_mtime > indexed_mtime


# время запуска и завершения,
# количество новых чанков,
# размер итогового индекса,
def log_success(operation_context):
    begin_date = operation_context.date
    begin_time = operation_context.begin_time
    end_time = operation_context.end_time
    log_path = os.path.join(logs_dir, f"log-{begin_date}.txt")
    with (open(log_path, "w", encoding="utf-8") as f):
        f.write(f"Начало работы: {begin_time}\n")
        f.write(f"Конец работы: {end_time}\n")
        if operation_context.chunks is not None and len(operation_context.chunks) != 0:
            f.write("Данные обновлены\n")
            f.write(f"Обработанные файлы: {len(operation_context.processed_files)}\n")
            for file in operation_context.processed_files:
                f.write(f"\t{file}\n")
            f.write(f"Количество новых чанков: {len(operation_context.chunks)}\n")
            if operation_context.index is not None:
                f.write(f"Размер итогового индекса: {operation_context.index}\n")
        else:
            f.write("Данные не обновлены\n")
            f.write("Нет новых или измененных файлов\n")


def log_error(operation_context):
    begin_date = operation_context.date
    begin_time = operation_context.begin_time
    end_time = operation_context.end_time
    log_path = os.path.join(logs_dir, f"log-{begin_date}.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Начало работы: {begin_time}\n")
        f.write(f"Конец работы: {end_time}\n")
        f.write("Ошибка при обновлении, данные не обновлены\n")
        f.write(f"{operation_context.error}\n")


class Context:
    def __init__(self):
        self.date = datetime.now().strftime("%Y-%m-%d")
        self.begin_time = datetime.now().strftime("%H:%M:%S")
        self.end_time = None
        self.chunks = None
        self.processed_files = None
        self.index = None
        self.error = None


if __name__ == "__main__":
    context = update_index()
    if context.error is None:
        log_success(context)
    else:
        log_error(context)
