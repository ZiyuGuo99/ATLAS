#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import re
from pathlib import Path

import httpx


PROMPT = """Below are a question, the standard answer (Ground Truth), and a model-generated answer.
Determine whether the model's answer is consistent with the standard answer.

Rules:
1. Multiple Choice: The selected option (A, B, C, D...) must match.
2. Math/Technical: Equivalent values are consistent.
3. Free-form: If the core meaning is the same, it is consistent.
4. If they are consistent, Correctness is 1; otherwise Correctness is 0.

[Question]: {question}
[Standard Answer]: {gt}
[Model Answer]: {pred}

Respond ONLY with JSON: {{"Correctness": 1}} or {{"Correctness": 0}}.
"""


def boxed(text: str | None) -> str | None:
    text = text or ""
    if "\\boxed{" not in text:
        return None
    start = [m.start() for m in re.finditer(r"\\boxed\{", text)][-1] + 7
    depth, out = 1, []
    for ch in text[start:]:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
        out.append(ch)
    return None


def clean_question(text: str | None) -> str:
    text = re.sub(r"<\|im_start\|>system.*?<\|im_end\|>", "", text or "", flags=re.S)
    text = re.sub(r"<\|im_start\|>user\n(<\|vision_start\|><\|vision_end\|>)?", "", text)
    return text.replace("<|im_end|>", "").replace("<|im_start|>assistant", "").strip()


def clean_reasoning(text: str | None) -> str:
    return re.sub(r"\\boxed\{.*?\}\s*$", "", text or "", flags=re.S).strip()


def load_samples(logs_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(logs_dir.glob("predictions*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
    return rows


def result_row(item: dict, pred: str | None, correct: float) -> dict:
    return {
        "qid": item.get("qids", ""),
        "question": clean_question(item.get("questions", "")),
        "reasoning": clean_reasoning(item.get("solutions", "")),
        "prediction": pred or "",
        "ground_truth": boxed(str(item.get("gts", ""))) or str(item.get("gts", "")),
        "judge_correctness": correct,
    }


async def judge_one(client: httpx.AsyncClient, item: dict, sem: asyncio.Semaphore, args) -> dict:
    pred = boxed(item.get("solutions", ""))
    if not pred:
        return result_row(item, pred, 0.0)

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": PROMPT.format(
            question=clean_question(item.get("questions", "")),
            gt=boxed(str(item.get("gts", ""))) or str(item.get("gts", "")),
            pred=pred,
        )}],
        "temperature": 0.0,
        "max_tokens": 50,
    }
    async with sem:
        resp = await client.post(args.api_url, json=payload, timeout=args.timeout)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        match = re.search(r'"Correctness"\s*:\s*([01])', text)
        if not match:
            raise RuntimeError(f"invalid judge response: {text!r}")
        return result_row(item, pred, float(match.group(1)))


def update_metrics(path: Path, benchmark: str, accuracy: float, ncorrect: float, ntotal: int) -> None:
    metrics = [{
        "benchmark": benchmark,
        "accuracy": accuracy,
        "ncorrect": ncorrect,
        "ntotal": ntotal,
    }]
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


async def run(args) -> None:
    logs_dir = args.result_dir / "logs"
    judged_path = logs_dir / "judged.jsonl"

    if judged_path.is_file() and not args.overwrite:
        rows = [json.loads(x) for x in judged_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    else:
        samples = load_samples(logs_dir)
        sem = asyncio.Semaphore(args.concurrency)
        async with httpx.AsyncClient(trust_env=False) as client:
            rows = await asyncio.gather(*(judge_one(client, row, sem, args) for row in samples))
        judged_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")

    ntotal = len(rows)
    ncorrect = sum(float(x.get("judge_correctness", 0.0)) for x in rows)
    accuracy = ncorrect / max(ntotal, 1)
    update_metrics(logs_dir / "metrics.json", args.result_dir.parent.name, accuracy, ncorrect, ntotal)
    print(f"[judge] score={accuracy:.4f} ({ncorrect:.0f}/{ntotal})")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path, nargs="?", help="Eval Output Directory")
    parser.add_argument("--api-url", default=os.getenv("JUDGE_API_URL", "http://127.0.0.1:8080/v1/chat/completions"))
    parser.add_argument("--server-url", default=os.getenv("JUDGE_SERVER_URL", "http://127.0.0.1:8080/v1/models"))
    parser.add_argument("--model", default=os.getenv("JUDGE_SERVED_MODEL", "JUDGE"))
    parser.add_argument("--concurrency", type=int, default=int(os.getenv("JUDGE_CONCURRENCY", "64")))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("JUDGE_TIMEOUT", "60")))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check-server", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.check_server:
        try:
            response = httpx.get(args.server_url, timeout=2, trust_env=False)
            response.raise_for_status()
        except Exception:
            raise SystemExit(1)
        raise SystemExit(0)
    if args.result_dir is None:
        raise SystemExit("result_dir is required unless --check-server is used")
    asyncio.run(run(args))
