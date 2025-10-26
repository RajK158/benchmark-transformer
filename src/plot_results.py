import pandas as pd
import matplotlib.pyplot as plt

# load results
df = pd.read_csv("results/results.csv")

# simple plot
plt.figure(figsize=(8,5))
for seq_len in df["sequence_length"].unique():
    sub = df[df["sequence_length"] == seq_len]
    plt.plot(sub["batch_size"], sub["avg_time_sec"], marker="o", label=f"Seq {seq_len}")

plt.title("Transformer Layer Benchmark on CPU")
plt.xlabel("Batch Size")
plt.ylabel("Average Time (seconds)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/benchmark_plot.png")
plt.show()
