# IGSD: Environment-Verified Hindsight Self-Distillation for Search Agents

## Repository structure

- `scripts/train_igsd.sh`: IGSD training launcher and main configuration.
- `examples/search_agent_rl/config/`: search-agent and tool configurations.
- `examples/search_agent_rl/preprocess_search_r1_dataset.py`: data preprocessing.
- `examples/search_agent_rl/local_dense_retriever/start_retrieval.sh`: local retriever launcher.
- `verl/trainer/main_ppo.py`: veRL training entry point.
- `verl/trainer/ppo/igsd_*.py`: IGSD training, verification, and distillation.
- `verl/experimental/agent_loop/tool_agent_loop.py`: search-agent rollout loop.
- `verl/tools/search_tool.py`: search tool interface.
- `tests/test_method_contract.py`: method configuration checks.

The `verl/` directory also contains the supporting veRL framework code.

## Training entry point

Run `bash scripts/train_igsd.sh` from the repository root. The launcher calls
`python3 -m verl.trainer.main_ppo` and requires `IGSD_MODEL_PATH`,
`IGSD_TRAIN_DATA`, `IGSD_VAL_DATA`, `IGSD_OUTPUT_DIR`, and
`IGSD_RETRIEVAL_URL` to be set for the local environment.

## Minimal launch example

From the repository root, first prepare the retriever index and corpus as
described in `examples/search_agent_rl/local_dense_retriever/README.md`.
With the retriever dependencies installed, start the service in one terminal:

```bash
bash examples/search_agent_rl/local_dense_retriever/start_retrieval.sh
```

Keep the retriever running. In a second terminal, start training:

```bash
export IGSD_MODEL_PATH=/path/to/model
export IGSD_TRAIN_DATA=/path/to/train.parquet
export IGSD_VAL_DATA=/path/to/validation.parquet
export IGSD_OUTPUT_DIR=/path/to/output
export IGSD_RETRIEVAL_URL=http://127.0.0.1:8000/retrieve

bash scripts/train_igsd.sh
```

Replace the placeholder paths with your local assets. For a different
retrieval endpoint, also set `IGSD_TOOL_CONFIG` to a tool configuration YAML
whose `retrieval_service_url` matches that endpoint. The default retriever and
training launchers both expose multiple GPUs; allocate non-overlapping devices
if they run on the same machine.
