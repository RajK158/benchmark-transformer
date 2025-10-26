import torch
import time
import pandas as pd
from tqdm import tqdm

# check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

# small transformer layer (like GPT-2)
layer = torch.nn.TransformerEncoderLayer(
    d_model=512, nhead=8, dim_feedforward=2048
).to(device)

# test settings
sequence_lengths = [512, 1024, 2048]
batch_sizes = [1, 8, 32]

results = []

for seq_len in sequence_lengths:
    for batch in batch_sizes:
        x = torch.randn(batch, seq_len, 512).to(device)
        torch.cuda.empty_cache() if device.type == "cuda" else None
        start = time.time()
        with torch.no_grad():
            for _ in tqdm(range(5), desc=f"B{batch}-S{seq_len}"):
                _ = layer(x)
        end = time.time()

        avg_time = (end - start) / 5
        results.append({
            "device": device.type,
            "batch_size": batch,
            "sequence_length": seq_len,
            "avg_time_sec": round(avg_time, 4)
        })

# save results
df = pd.DataFrame(results)
df.to_csv("results/results.csv", index=False)
print("\nResults saved to results/results.csv")
print(df)
