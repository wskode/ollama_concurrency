#!/usr/bin/env python3
"""
sweep.py – Find the concurrency sweet‑spot for an Ollama model
----------------------------------------------------------------
Runs the `run.py` benchmark repeatedly with an ascending list of concurrency
values (default: 1,2,4,8,16,32,64) and summarises when latency or error‑rate
starts to blow up.

Usage
~~~~
python3 sweep.py --model gemma3:27b             # default sweep list
python3 sweep.py --model llama3:8b -c 4 8 12 16 # custom list

Optional flags (forwarded to run.py):
  --requests/-n  number of requests per step
  --prompt       prompt text
  --tokens       num_predict per request
  --host         Ollama base URL
  --csv          master CSV (appends summaries)

Criteria for "trouble starts here" (tweakable):
* p95 latency ≥ *threshold* seconds  (default 30 s)
* error‑rate  ≥ 5 %
First concurrency level that violates either threshold is reported.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Dict

# Ensure run.py is importable
sys.path.append(str(Path(__file__).absolute().parent))
run = importlib.import_module("run")  # our sibling file


DEF_SWEEP = [1, 2, 4, 8, 16, 32, 64]


async def _bench_once(ns) -> Dict:
    summary, _results = await run._run_batch(ns)  # type: ignore[attr-defined]
    return summary


async def _sweep(args):
    trouble_at = None
    records: List[Dict] = []

    for conc in args.concurrency_list:
        ns = SimpleNamespace(
            model=args.model,
            host=args.host,
            concurrency=conc,
            requests=args.requests,
            prompt=args.prompt,
            tokens=args.tokens,
            csv=None,  # no per‑request CSV
        )
        print(f"\n▶️  Running {conc}‑way concurrency …", flush=True)
        summary = await _bench_once(ns)
        records.append(summary)

        ok = (
            summary["p95_latency"] is not None
            and summary["p95_latency"] < args.latency_threshold
            and summary["error_rate"] < args.error_threshold
        )
        if not ok and trouble_at is None:
            trouble_at = conc

    return trouble_at, records


def main():
    p = argparse.ArgumentParser(description="Sweep concurrency and detect when latency/error spikes.")
    p.add_argument("--model", required=True)
    p.add_argument("-c", "--concurrency-list", nargs="+", type=int, dest="concurrency_list", default=DEF_SWEEP)
    p.add_argument("-n", "--requests", type=int, default=50, help="Requests per concurrency level (default: 50)")
    p.add_argument("--prompt", default="Say 'hello, world!' in Korean.")
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--host", default="http://127.0.0.1:11434")
    p.add_argument("--latency-threshold", type=float, default=30.0, help="p95 latency threshold in seconds (default: 30)")
    p.add_argument("--error-threshold", type=float, default=0.05, help="Error‑rate threshold (default: 0.05)")
    p.add_argument("--csv", metavar="FILE", help="Append summary rows to this CSV")

    args = p.parse_args()

    trouble_at, records = asyncio.run(_sweep(args))

    print("\n――― Sweep Summary ―――")
    for rec in records:
        print(
            f"{rec['concurrency']:>3}c | p95={rec['p95_latency']:.2f}s | "
            f"err={rec['error_rate']*100:.1f}% | rps={rec['rps']:.2f}"
        )

    if trouble_at:
        print(f"\n⚠️  Performance degrades starting from ≧ {trouble_at} concurrent requests.")
    else:
        print("\n✅ No degradation detected within tested range.")

    # Optional: append to CSV file
    if args.csv:
        import csv as _csv

        fieldnames = records[0].keys()
        new_file = not Path(args.csv).exists()
        with open(args.csv, "a", newline="") as fp:
            writer = _csv.DictWriter(fp, fieldnames=fieldnames)
            if new_file:
                writer.writeheader()
            writer.writerows(records)
        print(f"Summary rows appended to {args.csv}")


if __name__ == "__main__":
    main()


