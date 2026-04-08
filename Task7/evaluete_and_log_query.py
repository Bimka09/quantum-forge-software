import json
from datetime import datetime
import os

LOG_FILE = os.path.join(os.path.dirname(__file__), "rag_logs.jsonl")

def evaluate_success(answer: str) -> bool:
    failure_patterns = [
        "я не знаю",
        "не найдено",
        "нет информации",
        "unknown",
        "sorry"
    ]

    if len(answer) < 20:
        return False

    answer_lower = answer.lower()
    if any(p in answer_lower for p in failure_patterns):
        return False

    return True

def evaluete_and_log(query, result, final_answer):
    sources = result.get("source_documents", [])

    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "query": query,
        "chunks_found": len(sources) > 0,
        "num_chunks": len(sources),
        "response_length": len(final_answer),
        "success_flag": evaluate_success(final_answer),
        "sources": [
            {
                "file": doc.metadata.get("file_name", "unknown"),
                "chunk": doc.metadata.get("chunk_index", "?")
            }
            for doc in sources
        ]
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

