import json
import logging
import os


def read_jsonl_files_for_prediction(
    input_files, conetxt_key="context", question_key="question", answers_key="answer", **kwargs
):
    if isinstance(input_files, str):
        input_files = [input_files]
    for f in input_files:
        if not os.path.exists(f):
            logging.warning("File %d does not exist, skipped.", f)
            continue
        with open(f, mode="rt", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                answers = data[answers_key]
                if isinstance(answers, str):
                    answers = [answers]
                answer = answers[0]
                instance = {"context": data[conetxt_key], "question": data[question_key], "answer": answer}
                yield instance
