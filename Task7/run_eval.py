import json
from datetime import datetime

from build_index import load_embeddings, load_vectorstore
from rag_pipeline import BGERetriever, load_llm, load_qa_chain

from bot import extract_final_answer
from evaluete_and_log_query import evaluete_and_log, evaluate_success

import os
LOG_FILE = os.path.join(os.path.dirname(__file__), "eval_logs.jsonl")
GOLDEN_SET_FILE = os.path.join(os.path.dirname(__file__), "golden_set.json")


def run_evaluation():
    with open(GOLDEN_SET_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    qa_chain = load_qa_chain(
        load_llm(),
        BGERetriever(vectorstore=load_vectorstore(load_embeddings()))
    )

    for item in dataset:
        query = item["question"]
        expected = item["expected"]

        result = qa_chain.invoke({"query": query})
        raw_answer = result["result"].strip()
        final_answer = extract_final_answer(raw_answer)

        success = evaluate_success(final_answer)

        evaluete_and_log(query, result, final_answer)

        # correctness
        if expected == "answer":
            correct = not success
        else:
            correct = success

        print("EVAL:", query, success)

        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "query": query,
            "expected": expected,
            "answer": final_answer,
            "success": success,
            "correct": correct,
            "num_sources": len(result.get("source_documents", []))
        }

        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    print("Evaluation completed.")

if __name__ == "__main__":
    run_evaluation()