from __future__ import annotations

if __package__ in {None, ""}:  # allow `python sub_src/run_exp1.py ...`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

from sub_src.apef_core import APEFConfig, run_apef
from sub_src.baselines import PRPConfig
from sub_src.datasets import DatasetKey, load_dataset_run
from sub_src.io_utils import write_json
from sub_src.llm import OpenAIChatConfig
from sub_src.metric import domain_metric
from sub_src.paths import ensure_results_dir
from sub_src.scenes import SCENE_ALIASES, SCENE_TARGET_WEIGHTS, SceneKey
from sub_src.splits import split_indices


def run_exp1(
    *,
    dataset: DatasetKey,
    scene: SceneKey,
    seed: int,
    steps: int,
    val_frac: float,
    test_frac: float,
    llm: str,
    openai_model: str,
    openai_model_policy_extract: str | None,
    openai_model_policy_score: str | None,
    temperature: float | None,
    normalize: bool,
    warmup_steps: int,
    policy_extract_every: int,
    policy_eval_runs: int,
    policy_accept_frac: float,
    require_positive_first_policy: bool,
    skip_prp: bool,
    prp_k: int,
    prompt_ndigits: int,
    out_base: Path,
) -> Path:
    scene = SCENE_ALIASES[scene]
    ds = load_dataset_run(dataset)
    gt = ds.ground_truth
    preds = ds.predictions

    target_w = SCENE_TARGET_WEIGHTS[scene]
    target_scores = [
        float(domain_metric(gt, preds[i], target_w, normalize=normalize)) for i in range(preds.shape[0])
    ]

    train_idx, val_idx, test_idx = split_indices(
        preds.shape[0], seed=seed, val_frac=val_frac, test_frac=test_frac
    )

    openai_cfg = OpenAIChatConfig(model=openai_model, temperature=temperature) if llm == "openai" else None
    openai_cfg_policy_extract = (
        OpenAIChatConfig(model=(openai_model_policy_extract or openai_model), temperature=temperature)
        if llm == "openai"
        else None
    )
    openai_cfg_policy_score = (
        OpenAIChatConfig(model=(openai_model_policy_score or openai_model), temperature=temperature)
        if llm == "openai"
        else None
    )
    prp_cfg = PRPConfig(
        enabled=not skip_prp,
        k=prp_k,
        ndigits=prompt_ndigits,
        cache_dir=out_base / "_cache",
    )
    apef_cfg = APEFConfig(
        steps=steps,
        warmup_steps=warmup_steps,
        policy_extract_every=policy_extract_every,
        policy_eval_runs=policy_eval_runs,
        policy_accept_frac=policy_accept_frac,
        require_positive_first_policy=require_positive_first_policy,
        prompt_ndigits=prompt_ndigits,
    )

    result = run_apef(
        exp_name="exp1",
        dataset_key=dataset,
        feature_name=ds.feature_name,
        scene=scene,
        ground_truth=gt,
        predictions=preds,
        ts_labels=ds.ts_labels,
        target_scores=target_scores,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        llm_mode=llm,  # type: ignore[arg-type]
        openai_cfg=openai_cfg,
        openai_cfg_policy_extract=openai_cfg_policy_extract,
        openai_cfg_policy_score=openai_cfg_policy_score,
        normalize=normalize,
        prp_cfg=prp_cfg,
        out_base=out_base,
        seed=seed,
        cfg=apef_cfg,
    )

    write_json(result.run_dir / "target_weights.json", target_w.__dict__)
    return result.run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exp1 (target-weight imitation) in sub_src.")
    parser.add_argument("--dataset", required=True, choices=["co2", "gpp", "camels", "xmethanewet"])
    parser.add_argument("--scene", required=True, choices=sorted(SCENE_ALIASES.keys()))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--policy-extract-every", type=int, default=1)
    parser.add_argument("--policy-eval-runs", type=int, default=1)
    parser.add_argument("--policy-accept-frac", type=float, default=0.6)
    parser.add_argument("--allow-nonpositive-first-policy", action="store_true")
    parser.add_argument("--val-frac", type=float, default=0.25)
    parser.add_argument("--test-frac", type=float, default=0.0)
    parser.add_argument("--llm", choices=["none", "openai"], default="none")
    parser.add_argument("--openai-model", default="gpt-5-nano")
    parser.add_argument("--openai-model-policy-extract", default=None)
    parser.add_argument("--openai-model-policy-score", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--skip-prp", action="store_true")
    parser.add_argument("--prp-k", type=int, default=2)
    parser.add_argument("--prompt-ndigits", type=int, default=4)
    args = parser.parse_args()

    out_base = ensure_results_dir()
    run_dir = run_exp1(
        dataset=args.dataset,
        scene=args.scene,
        seed=args.seed,
        steps=args.steps,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        llm=args.llm,
        openai_model=args.openai_model,
        openai_model_policy_extract=args.openai_model_policy_extract,
        openai_model_policy_score=args.openai_model_policy_score,
        temperature=args.temperature,
        normalize=not args.no_normalize,
        warmup_steps=args.warmup_steps,
        policy_extract_every=args.policy_extract_every,
        policy_eval_runs=args.policy_eval_runs,
        policy_accept_frac=args.policy_accept_frac,
        require_positive_first_policy=not args.allow_nonpositive_first_policy,
        skip_prp=args.skip_prp,
        prp_k=args.prp_k,
        prompt_ndigits=args.prompt_ndigits,
        out_base=out_base,
    )
    print(str(run_dir))


if __name__ == "__main__":
    main()
