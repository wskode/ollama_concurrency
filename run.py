#!/usr/bin/env python3
"""
run.py – Lightweight concurrency benchmark for an Ollama server.

Usage examples
--------------
# Quick smoke‑test with defaults (32 parallel requests, 100 total)
python3 run.py --model llama3:8b

# Custom settings, save detailed per‑request metrics to CSV
python3 run.py \
    --model gemma3:27b \
    --concurrency 64 \
    --requests 1000 \
    --prompt "Explain TCP three‑way handshake" \
    --tokens 256 \
    --csv results_gemma3_64c.csv

Summary statistics (p50 / p95 latency, requests‑per‑second, error‑rate) are
printed to stdout.  If --csv is supplied, every request’s latency, token count
and error status are written to that file for later analysis (e.g. Excel,
Grafana, pandas).

Dependencies
------------
python -m pip install "httpx>=0.27,<1.0"

Tested with Python 3.10+.
"""
import argparse
import asyncio
import csv
import statistics
import time
from datetime import datetime
from typing import List, Dict

import httpx

# ――― Internal helpers ――― ---------------------------------------------------
async def _one_request(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    sem: asyncio.Semaphore,
    results: List[Dict],
) -> None:
    """Send one /api/generate POST and record latency & outcome."""
    started = time.perf_counter()
    async with sem:
        try:
            resp = await client.post(url, json=payload, timeout=None)
            latency = time.perf_counter() - started
            entry = {"latency": latency, "status": resp.status_code}

            if resp.status_code == 200:
                data = resp.json()
                # Ollama returns durations in µs; convert to seconds if present
                dur_us = data.get("total_duration")
                if dur_us is not None:
                    entry["total_duration"] = dur_us / 1e6
                entry["tokens"] = data.get("eval_count")
            else:
                entry["error"] = resp.text[:200]
        except Exception as exc:  # network/timeout etc.
            latency = time.perf_counter() - started
            entry = {
                "latency": latency,
                "status": "exception",
                "error": repr(exc)[:200],
            }
        results.append(entry)


async def _run_batch(args) -> (dict, List[Dict]):
    """Launch *args.requests* POSTs limited by *args.concurrency*."""
    url = args.host.rstrip("/") + "/api/generate"
    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "stream": False,
        "options": {"num_predict": args.tokens},
    }

    sem = asyncio.Semaphore(args.concurrency)
    results: List[Dict] = []
    tasks = []

    async with httpx.AsyncClient(timeout=None) as client:
        batch_start = time.perf_counter()
        for _ in range(args.requests):
            tasks.append(
                asyncio.create_task(_one_request(client, url, payload, sem, results))
            )
        await asyncio.gather(*tasks)
    batch_elapsed = time.perf_counter() - batch_start

    # ――― Aggregate stats ―――
    ok_latencies = [r["latency"] for r in results if r["status"] == 200]
    p50 = statistics.median(ok_latencies) if ok_latencies else None
    p95 = (
        statistics.quantiles(ok_latencies, n=100)[94] if len(ok_latencies) >= 100 else None
    )
    error_rate = 1 - (len(ok_latencies) / len(results)) if results else 1.0
    rps = len(results) / batch_elapsed if batch_elapsed > 0 else 0

    summary = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": args.model,
        "host": args.host,
        "concurrency": args.concurrency,
        "requests": args.requests,
        "prompt_len": len(args.prompt),
        "tokens": args.tokens,
        "p50_latency": p50,
        "p95_latency": p95,
        "rps": rps,
        "error_rate": error_rate,
        "total_time": batch_elapsed,
    }
    return summary, results


# ――― CLI entrypoint ――― ------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark how many concurrent requests an Ollama server can handle."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model identifier as known to Ollama, e.g. llama3:8b or gemma:27b",
    )
    parser.add_argument(
        "--host",
        default="http://127.0.0.1:11434",
        help="Base URL of the Ollama server (default: %(default)s)",
    )
    parser.add_argument("--concurrency", "-c", type=int, default=32, help="Parallel requests")
    parser.add_argument(
        "--requests",
        "-n",
        type=int,
        default=100,
        help="Total number of requests to send (default: %(default)s)",
    )
    parser.add_argument(
        "--prompt",
        default="Say 'hello, world!' in Korean.",
        help="Prompt text to use (default: simple hello-world)",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=128,
        help="Maximum tokens per response (num_predict).",
    )
    parser.add_argument(
        "--csv",
        metavar="FILE",
        help="Write per‑request detailed metrics to CSV file.",
    )

    args = parser.parse_args()

    summary, results = asyncio.run(_run_batch(args))

    print("\n――― Summary ―――")
    for k, v in summary.items():
        print(f"{k:>15}: {v}")

    if args.csv:
        fieldnames = sorted({key for r in results for key in r})
        with open(args.csv, "w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nDetailed metrics written to {args.csv}")


if __name__ == "__main__":
    main()

