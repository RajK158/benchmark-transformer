![Status](https://img.shields.io/badge/Project%20Status-Completed-brightgreen)

**Benchmarking Transformer Layers on Different Hardware**

**Author: 1. Raj Kundur – MS CS Student, CSULB,
        2. Aaradhya Partap - MS CS Student, CSULB**

📘 Project Overview

This project tests how fast a Transformer layer (a small part of models like GPT) runs on different hardware types such as CPU, GPU, and TPU.
It measures how long it takes to process data and how performance changes with bigger input sizes.

Since this was run on a laptop (Intel Core i3 CPU), the current results show CPU-only performance.
GPU/TPU results are discussed in the presentation slides.

⚙️ What is Benchmarking?

Benchmarking means comparing performance.
Here, we compare how fast one Transformer layer runs on various hardware by measuring:

Average time (seconds)

Batch size (how many samples processed at once)

Sequence length (length of input data)

🧩 Hardware Tested

CPU: Intel Core i3 10th Gen (Laptop)

GPU / TPU: Results referenced from presentation (H100, A100, TPU v4, MI300X)

🧠 Key Idea: CPU vs GPU vs TPU
Type	Description	Example
CPU	General-purpose processor with few cores. Great for basic tasks.	Intel Core i3, i7
GPU	Thousands of smaller cores specialized for parallel math. Best for AI models.	NVIDIA A100, H100
TPU	Custom chip built by Google just for AI matrix operations.	Google TPU v4
🧪 How to Run
1️⃣ Create Environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

2️⃣ Run Benchmark
python src/benchmark.py

3️⃣ Plot Results
python src/plot_results.py

Output Files

results/results.csv → Timing results

results/benchmark_plot.png → Graph of latency vs batch size

📊 Example Result (CPU)
Batch Size	Seq Length	Avg Time (s)
1	512	0.0793
8	1024	0.9467
32	2048	8.3274

Larger sequences or batches take more time because the CPU processes them sequentially.

🧾 Future Improvements

Add GPU and TPU benchmarks (using Google Colab or cloud).

Compare energy usage and cost per performance.

Automate plots for hardware comparison.

📚 References

Team Presentation Slides (Advanced Computer Architecture – CSULB)

PyTorch Documentation: https://pytorch.org

NVIDIA, AMD, and Google hardware whitepapers
