# Ollama Concurrency Benchmark

> Lightweight Python script (`run.py`) to measure how many concurrent `POST /api/generate` requests your Ollama server can sustain, and to record latency / throughput metrics for later analysis.

---

## 📦 Requirements

| Dependency | Version     | Install                                   |
| ---------- | ----------- | ----------------------------------------- |
| Python     | 3.10+       | <sup>`sudo apt‑get install python3`</sup> |
| **httpx**  | ≥ 0.27,<1.0 | `pip install "httpx>=0.27,<1.0"`          |

No other third‑party packages are needed for the benchmark itself.

> **Tip (virtualenv)**  Create an isolated env so different experiments don’t clash:
>
> ```bash
> python -m venv .venv && source .venv/bin/activate
> pip install "httpx>=0.27,<1.0"
> ```

---

## 🚀 Quick Start

```bash
# 32 concurrent requests, 100 total
python3 run.py --model llama3:8b

# 64-way concurrency, 1 000 requests, write every result to CSV
python3 run.py \
    --model gemma3:27b \
    --concurrency 64 \
    --requests 1000 \
    --csv results_gemma3_64c.csv
```

You will see a **Summary** block like:

```
       timestamp: 2025-06-04T07:12:55Z
          model: gemma3:27b
           host: http://127.0.0.1:11434
     concurrency: 64
        requests: 1000
      prompt_len: 27
          tokens: 128
     p50_latency: 1.83
     p95_latency: 3.25
             rps: 34.1
       error_rate: 0.002
      total_time: 29.3
```

If `--csv` is supplied, each request’s metrics (`latency`, `status`, `tokens`, `total_duration`, `error`) are appended to the specified file.

---

## ⚙️ Command‑line Options

| Flag                | Description                                                        | Default                          |
| ------------------- | ------------------------------------------------------------------ | -------------------------------- |
| `--model`           | Model ID exactly as loaded in Ollama (`llama3:8b`, `gemma:27b`, …) | *required*                       |
| `--host`            | Base URL of the Ollama server                                      | `http://127.0.0.1:11434`         |
| `-c, --concurrency` | Max simultaneous requests (async semaphore size)                   | `32`                             |
| `-n, --requests`    | Total requests to send                                             | `100`                            |
| `--prompt`          | Prompt text                                                        | `Say 'hello, world!' in Korean.` |
| `--tokens`          | `num_predict` (max tokens in response)                             | `128`                            |
| `--csv <file>`      | Path to save detailed per‑request metrics                          | *(skip)*                         |

---

## 🔬 Analysing the Results

### 1. With **pandas** + matplotlib

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results_gemma3_64c.csv")
print(df.describe(percentiles=[0.5, 0.95]))  # p50/p95 latency etc.

# Histogram of latencies
plt.figure()
df["latency"].plot.hist(bins=50)
plt.xlabel("Latency (s)")
plt.title("Latency distribution – gemma3:27b, 64‑concurrency")
plt.show()
```

Further ideas:

* Bucket by `status` to isolate error outliers.
* Compute throughput per second with `df.groupby(df.index // 64).size()` when concurrency is constant.

### 2. With **Grafana**

There are two quick routes:

<details>
<summary>Option A – CSV data‑source plugin (zero backend)</summary>

1. In Grafana ➜ **Connections → Data sources → CSV**, install the official CSV plugin.
2. Point it at your CSV file (local file path or HTTP‑served directory).
3. Use a table panel to preview, then build graphs:

   * *Time field* → leave empty (use row number as X‑axis) **or** create a derived column `timestamp` in your CSV (UTC isoformat).
   * Plot `latency` line, overlay p95 via Transform → **Percentile**.

</details>

<details>
<summary>Option B – Import into Prometheus via Pushgateway</summary>

1. Run a Pushgateway (`docker run -d -p 9091:9091 prom/pushgateway`).
2. After each benchmark, convert CSV to Prometheus Exposition format and `curl` to `localhost:9091/metrics/job/ollama_bench`.
3. Configure Prometheus → Grafana → build dashboards (latency, RPS, error rate) with alert rules (`p95_latency > 5s for 5m`).

</details>

Either way you can create:

* **Latency heat‑maps** over time.
* **p95 latency vs concurrency** by adding repeated runs as separate series.
* **Error rate panels** with `status != 200` filter.

### 3. Other tools

| Tool                      | Why use it?                                                                                         |
| ------------------------- | --------------------------------------------------------------------------------------------------- |
| **Excel / Google Sheets** | Quick pivot on `status` and percentile functions.                                                   |
| **DuckDB**                | `duckdb-shell` → `SELECT percentile_cont(0.95) WITHIN GROUP (ORDER BY latency) FROM 'results.csv';` |
| **Jupyter Notebook**      | Interactive; run multiple model/concurrency CSVs, merge, and plot comparative charts.               |

---

## 🧩 Extending the Benchmarker

* **Custom prompt file**: read prompts line‑by‑line and random‑choice per request.
* **Multiple model sweep**: loop over a list of models/concurrency values.
* **Progress bar**: wrap the `asyncio.gather` loop with `tqdm.asyncio.tqdm_asyncio`.
* **Automatic stop‑on‑error**: abort the run if error rate crosses a threshold.

PRs welcome 🙂

---

## 📝 License

MIT (see `LICENSE` file).

