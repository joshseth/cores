# Transformers Converge to Invariant Algorithmic Cores

This repository contains code for analyses and plots for the paper **Transformers Converge to Invariant Algorithmic Cores**.

## Setup

From the repository root:

```bash
python -m pip install -e .
```

## Markov experiment

Run:

```bash
python markov.py
```

Main plot output:

```text
experiments/markov_chain/results/plots/Markov_fig.pdf
```

## Modular addition experiment

Full paper run:

```bash
python modadd.py
```

Main plot outputs:

```text
experiments/modular_addition/mod_add_fig1.pdf
experiments/modular_addition/mod_add_fig2.pdf
```

## Grokking sweep

Full paper run:

```bash
python grok_sweep.py
```

Main plot outputs:

```text
experiments/grok_sweep/grok_sweep_two_panel.pdf
experiments/grok_sweep/grok_sweep_two_panel.png
```

## Subject-verb agreement experiment

The SVA wrapper runs the GPT-2 family experiments:

- GPT-2 Small
- GPT-2 Medium
- GPT-2 Large

and then generates the combined plot.

```bash
python sva.py quick
```

Main plot output:

```text
experiments/sva/plots/SVA_figure.pdf
```

Logs are written to:

```text
experiments/sva/logs/
```

The same SVA script can also be run on other models, including LLaMA-3.1-8B/9B-style models and Qwen2.5-32B, by changing the model name and run name, for example:

```bash
python sva.py run --model-name meta-llama/Meta-Llama-3.1-8B --run-name llama31 --seed 0 --run-layer-sweep
python sva.py run --model-name google/gemma-2-9b --run-name gemma2 --seed 0 --run-layer-sweep
python sva.py run --model-name Qwen/Qwen2.5-32B --run-name qwen25_32b --seed 0 --run-layer-sweep
```

## Outputs

Generated results and figures are written under:

```text
experiments/
```
