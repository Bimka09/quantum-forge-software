import json
import time

import chromadb
from chunks import process_directory  
from langchain_community.embeddings import HuggingFaceEmbeddings

# Имя коллекции в ChromaDB
COLLECTION_NAME = "yandex_rag_bot"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# Загружаем модель для генерации эмбеддингов
def load_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 16,
        },
    )

# Инициализация клиента ChromaDB
chroma_db_client = chromadb.PersistentClient(path="./chromadb/")

# Создаём или получаем коллекцию
collection = chroma_db_client.get_or_create_collection(name=COLLECTION_NAME)

def generate_and_store_embeddings(chunks):
    model = load_embeddings()

    texts = [chunk['text'] for chunk in chunks]
    embeddings = model.embed_documents(texts)

    ids = []
    metadatas = []
    documents = []

    for i, chunk in enumerate(chunks):
        ids.append(f"{chunk['source']}_{i}")
        documents.append(chunk['text'])
        metadatas.append({
            "source": chunk.get("source", "unknown"),
            "last_updated": chunk.get("last_updated", 0.0),
            "chunk_id": chunk.get("chunk_index", i),
            "token_count": chunk.get("token_count", 0)
        })

    collection.add(
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
        documents=documents
    )

    print(f"✅ Добавлено чанков: {len(embeddings)}")
# Считать все файлы и даты обновления из индекса
def get_file_index():
    if collection.count() == 0:
        return {}
    else:
        file_index = {}
        for item in collection.get(include=['metadatas'])['metadatas']:
            source = item.get('source')
            last_updated = item.get('last_updated')

            if source is None:
                continue

            # если поле отсутствует — считаем файл "старым"
            if last_updated is None:
                last_updated = 0.0

            file_index[source] = last_updated

        return file_index