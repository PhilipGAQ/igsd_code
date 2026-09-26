# IGSD: Environment-Verified Hindsight Self-Distillation

This repository contains the training implementation for IGSD, built on the
[veRL](https://github.com/volcengine/verl) framework and the SearchAgent-Zero
search-agent workflow. Upstream license and attribution are retained in
`LICENSE` and `Notice.txt`. It includes the IGSD training entry point,
retrieval interface, and preprocessing code, not checkpoints, datasets,
evaluation outputs, or experimental ablation launchers.

## Method implemented

For a failed rollout with a successful sibling, IGSD uses the sibling's search
queries and terminal score as a hindsight hint for a frozen self-teacher. It
selects the failed rollout's last valid search action and compares the teacher
top-1 query token with the token sampled by the student. The same unprivileged
frozen student greedily completes each candidate into a query; both queries
are executed with the same retriever. The paired difference of
reference-answer likelihood gains over shared random-document controls
provides an evidence-utility contrast. A detached positive-only gate weights
a two-token candidate-pair Jensen–Shannon distillation term alongside GRPO.
Neither teacher hints nor verification branches are used at inference.

There is **no positive-likelihood-sign prefilter**. Both positive- and
negative-shift disagreements can reach environment verification. The training
configuration still bounds the number of paired executions per query to
`floor(number_of_query_tokens × budget_ratio) / 2` pairs (budget ratio 1.0);
when that bound is active, pairs are prioritized by the absolute detached
teacher–student log-odds shift. This is a *compute-budget ordering*, not the
acceptance criterion: only positive executed evidence gain activates the loss.
It is retained for fidelity to the original main training configuration.

## Requirements and input assets

- Python 3.10+, PyTorch/CUDA, veRL dependencies (see `pyproject.toml` and
  `requirements.txt`), and a vLLM-compatible 3B or 7B Qwen2.5-Instruct model.
- Search-R1-format NQ/HotpotQA training and validation Parquet files with
  `prompt`, `reward_model.ground_truth`, and `extra_info.tools_kwargs` fields.
  The included [preprocessor](examples/search_agent_rl/preprocess_search_r1_dataset.py)
  shows the expected schema; review its source dataset and validation split
  before applying it to new data.
- A compatible local E5 retriever over the December 2018 Wikipedia corpus,
  exposing POST `/retrieve` and the `queries`/`topk` request schema. The
  [retriever example](examples/search_agent_rl/local_dense_retriever/README.md)
  documents the separate service. Its index/corpus and weights are not bundled.
- A machine with enough accelerator memory for eight-GPU, long-context
  training. Availability of those assets and an end-to-end run are **not**
  asserted by this release.

## Train

Set paths appropriate to your own machine. The launcher deliberately has no
private paths or server-specific defaults; it requires an already running
retrieval service and does not start or download anything implicitly.

```bash
export IGSD_MODEL_PATH=/path/to/Qwen2.5-3B-Instruct
export IGSD_TRAIN_DATA=/path/to/train_search_r1.parquet
export IGSD_VAL_DATA=/path/to/test_search_r1.parquet
export IGSD_OUTPUT_DIR=/path/to/output
export IGSD_RETRIEVAL_URL=http://127.0.0.1:8000/retrieve
bash scripts/train_igsd.sh
```

The launcher sets the published primary configuration: GRPO, batch size 256,
5 rollouts per prompt, 400 steps, 4096/3000 prompt/response lengths, 6 search
turns, 3 passages per query, last-search supervision, 3 random controls,
candidate-pair JSD, `β=5`, and auxiliary coefficient `λ=0.01`. Override
`IGSD_N_GPUS_PER_NODE` and device visibility only after verifying the
resulting effective batch/memory configuration. The script does not include
baseline or ablation launchers.

## Limitations and verification

This is a curated source release, not a packaged model or a claim of a
fresh end-to-end reproduction. The original paper's 400-step curves,
terminal-pair evaluations, and dataset artifacts are not reproduced here.
Use the tests under `tests/` for CPU-only checks and validate the complete
training/evaluation workflow on the target machine and data before citing new
numerical results. The paired verification budget may skip disagreements
when exceeded; report its effect if you change the budget.

The underlying veRL trainer retains compatibility code for optional
algorithms and historical auxiliary objectives; only the main IGSD
configuration is exposed by the launcher. Those inactive compatibility paths
are not part of the method described above.
