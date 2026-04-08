import json

def analyze_logs(file="eval_logs.jsonl"):
    total = 0
    correct = 0
    no_answer_fail = 0

    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            total += 1

            if data["correct"]:
                correct += 1

            if data["expected"] == "answer" and not data["success"]:
                no_answer_fail += 1

    print(f"Accuracy: {correct / total:.2f}")
    print(f"Missed answers: {no_answer_fail}")