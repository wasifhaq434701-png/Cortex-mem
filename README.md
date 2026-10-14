# PhoneAI — what actually makes a large language model fit on a phone

Research code for running capable LLMs on memory-bandwidth-bound devices (target: iPhone 18 Pro, A20 Pro,
12 GB). All experiments below ran on a MacBook Pro M4 Max (36 GB) with `mlx-lm`; on-device measurements
are next.

**Question.** Can per-request *conditional computation* — choosing which parameters to keep resident or
which module to load, from a cheap signal about the request — buy big-model quality inside a small RAM
budget?

**Answer so far: not at the parameter level.** Three experiments, each against a fair dense or shared
baseline, all negative:

| # | Idea | Setup | Result |
|---|---|---|---|
| 1 | Task-predicted MoE expert residency | Qwen3-30B-A3B, per-task expert usage, 25 prompts × 5 tasks | Locality is real (task-predicted hit rate 64% vs 49% global at 32/128 experts), but misses still cost ~210 MB of flash reads per token |
| 2 | Route only among resident experts (no flash reads) | Router hard-masked + 7 training-free rescue variants, GSM8K / HumanEval | Collapses well before phone budgets: 94% → 10% GSM8K at 7 GB of resident weights; a dense 8B in 4.6 GB scores 89% |
| 3 | Dense core + per-task LoRA modules picked per request | Qwen3-4B, math / code / notes adapters vs one multi-task LoRA, 4 evals, paired bootstrap | Task routing is within noise of a single shared LoRA at equal per-request cost **and** at equal storage |

Experiment 3 in numbers (round 2, self-distilled training data; 95% CIs in `results/exp3.json`):

| config | GSM8K | HumanEval pass@1 | personal-notes EM | DROP EM |
|---|---|---|---|---|
| Qwen3-4B, no adapter | 88.0 | 77.4 | 57.5 | 61.3 |
| oracle task adapter | 88.0 | 76.2 | 69.0 | 66.3 |
| embedding-routed adapter | 88.0 | 76.2 | 69.0 | 64.0 |
| one multi-task LoRA, rank 8 | 86.4 | 80.5 | 67.0 | 66.0 |
| one multi-task LoRA, rank 24 | 88.4 | 78.0 | 67.5 | 64.3 |

Two side findings: fine-tuning on dataset reference answers *damages* a strong instruct model (round 1,
kept in `results/exp3_round1/`), and sentence-embedding routing misroutes 27% of DROP questions
(number-heavy passages look like math).

**Where this points.** Per-request selection is necessary, not optional, for *per-user* state — one shared
model cannot hold every user's private data. Next: tiered on-device memory (text nodes + embeddings +
graph → cached KV for hot nodes → small style adapters), selected into a fixed RAM budget by Cortex
([Ul Haq 2026, DOI 10.5281/zenodo.22705201](https://doi.org/10.5281/zenodo.22705201)), evaluated on
LongMemEval and LoCoMo, then measured on the iPhone.

## Reproduce

```bash
python experiments/expert_locality.py                         # Exp 1
python experiments/routing_quality.py --model mlx-community/Qwen3-30B-A3B-4bit   # Exp 2
python experiments/routing_quality.py --model mlx-community/Qwen3-8B-4bit
python experiments/task_adapters.py check                     # Exp 3 self-checks
python experiments/task_adapters.py distill && python experiments/task_adapters.py train
python experiments/task_adapters.py eval && python experiments/task_adapters.py report
```

Exp 3 takes ~11 h on an M4 Max; generations are cached in `results/exp3_gen/`, so `report` reruns from the
committed results without a GPU. Adapter weights are not in git (`adapters/`, ~700 MB, available on request).

## Report

`report/report.pdf` (v1, October 2026) is the write-up of these three experiments; `report/report.tex` and
`report/make_figures.py` rebuild it from the committed results.

## Layout

- `experiments/` — one script per experiment; `notes_synth.py` generates the personal-notes benchmark
- `data/exp3/` — training sets (round 2, self-distilled); `data/exp3_round1/` — reference-answer sets
- `results/` — per-item scores, raw generations, training logs, summary JSON for every reported number
- `report/` — the technical report and the script that draws its figures

Data in `data/` and `results/` is derived from public benchmarks under their own licenses: GSM8K (MIT),
HumanEval (MIT), MBPP (CC BY 4.0) and DROP (CC BY-SA 4.0). The synthetic personal-notes items are generated
by `experiments/notes_synth.py`.
