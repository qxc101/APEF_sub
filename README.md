# `Implementation of APEF`
This is the official implementation for paper LLM-based Evaluation Policy Extraction for Ecological Modeling. 
## Quick start

Set up environment:

```bash
conda env create -f domain_metric/sub_src/domain_copy_env.yml
```

Run all experiments in the paper
```bash
OPENAI_API_KEY=... 
python -m sub_src.run_all --llm openai --openai-model gpt-5-nano
```

## Speed knobs

- `--skip-prp` (PRP baseline is LLM-expensive)
- `--policy-extract-every N` (policy scoring dominates calls)
- `--val-frac-exp1 F` / `--val-frac-exp2 F` (policy scoring is 1 call per val TS)

## Use different models for policy

- `--openai-model-policy-extract` (policy generation; higher quality)
- `--openai-model-policy-score` (policy scoring; faster)

Both default to `--openai-model` if omitted.


