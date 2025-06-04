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