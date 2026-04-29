# Aroviq Performance Benchmarks

This directory contains the actual scripts used to generate the latency metrics and performance claims cited in the Aroviq documentation.

## Running the Benchmark

The included `run_comparison.py` script executes a direct performance comparison between Aroviq's Tier 0 (Regex/Symbolic Rules) engine and various Tier 1 LLM providers.

### Prerequisites

To fully test the cloud-based Tier 1 models, ensure you export your API keys before running the suite:

```bash
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="AIza..."
```

### Execution

Run the script from the root of the repository:

```bash
poetry run python benchmarks/run_comparison.py
```

### Reproducing the "8,000x Faster" Claim

Our claim that Tier 0 validation is "up to 8,000x faster" than an LLM is a reflection of local CPU execution for regex evaluations versus the round-trip latency of cloud models or the inference time of local LLMs. 

*   **Tier 0 (Regex):** ~0.15ms latency
*   **Tier 1 (LLM Cloud):** ~1200ms latency

This speedup factor calculates `$ \frac{1200 \text{ ms}}{0.15 \text{ ms}} = 8000 $`. 
Running the benchmark suite above on your local machine will demonstrate this architectural difference firsthand.
