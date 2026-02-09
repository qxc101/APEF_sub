from __future__ import annotations

if __package__ in {None, ""}:  # allow `python sub_src/run_all.py ...`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import time
from pathlib import Path

from sub_src.paths import ensure_results_dir
from sub_src.run_exp1 import run_exp1
from sub_src.run_exp2 import run_exp2
from sub_src.scenes import SCENE_TARGET_WEIGHTS


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full exp1/exp2 matrix.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--policy-extract-every", type=int, default=1)
    parser.add_argument("--policy-eval-runs", type=int, default=1)
    parser.add_argument("--policy-accept-frac", type=float, default=0.6)
    parser.add_argument("--allow-nonpositive-first-policy", action="store_true")
    parser.add_argument("--val-frac-exp1", type=float, default=0.5)
    parser.add_argument("--val-frac-exp2", type=float, default=0.5)
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
    parser.add_argument("--skip-exp2", action="store_true")
    parser.add_argument("--skip-exp1", action="store_true")
    args = parser.parse_args()

    out_base: Path = ensure_results_dir()
    normalize = not args.no_normalize

    datasets = ["co2", "gpp", "camels", "xmethanewet"]
    total = (0 if args.skip_exp1 else (len(datasets) * len(SCENE_TARGET_WEIGHTS))) + (0 if args.skip_exp2 else len(datasets))
    done = 0
    run_dirs: list[Path] = []

    if not args.skip_exp1:
        for dataset in datasets:
            for scene in SCENE_TARGET_WEIGHTS.keys():
                done += 1
                print(f"[{done}/{total}] exp1 {dataset} {scene}")
                run_dir = run_exp1(
                    dataset=dataset,  # type: ignore[arg-type]
                    scene=scene,  # type: ignore[arg-type]
                    seed=args.seed,
                    steps=args.steps,
                    warmup_steps=args.warmup_steps,
                    policy_extract_every=args.policy_extract_every,
                    policy_eval_runs=args.policy_eval_runs,
                    policy_accept_frac=args.policy_accept_frac,
                    require_positive_first_policy=not args.allow_nonpositive_first_policy,
                    val_frac=args.val_frac_exp1,
                    test_frac=args.test_frac,
                    llm=args.llm,
                    openai_model=args.openai_model,
                    openai_model_policy_extract=args.openai_model_policy_extract,
                    openai_model_policy_score=args.openai_model_policy_score,
                    temperature=args.temperature,
                    normalize=normalize,
                    skip_prp=args.skip_prp,
                    prp_k=args.prp_k,
                    prompt_ndigits=args.prompt_ndigits,
                    out_base=out_base,
                )
                run_dirs.append(run_dir)

    if not args.skip_exp2:
        for dataset in datasets:
            done += 1
            print(f"[{done}/{total}] exp2 {dataset}")
            run_dir = run_exp2(
                dataset=dataset,  # type: ignore[arg-type]
                seed=args.seed,
                steps=args.steps,
                warmup_steps=args.warmup_steps,
                policy_extract_every=args.policy_extract_every,
                policy_eval_runs=args.policy_eval_runs,
                policy_accept_frac=args.policy_accept_frac,
                require_positive_first_policy=not args.allow_nonpositive_first_policy,
                val_frac=args.val_frac_exp2,
                test_frac=args.test_frac,
                llm=args.llm,
                openai_model=args.openai_model,
                openai_model_policy_extract=args.openai_model_policy_extract,
                openai_model_policy_score=args.openai_model_policy_score,
                temperature=args.temperature,
                normalize=normalize,
                skip_prp=args.skip_prp,
                prp_k=args.prp_k,
                prompt_ndigits=args.prompt_ndigits,
                out_base=out_base,
            )
            run_dirs.append(run_dir)

    # Aggregate report
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_rows: list[dict[str, object]] = []
    for rd in run_dirs:
        try:
            summary = json.loads((Path(rd) / "summary.json").read_text(encoding="utf-8"))
        except Exception:
            continue

        baselines = summary.get("baselines_spearman", {})
        val_base = baselines.get("val", {}) if isinstance(baselines, dict) else {}
        test_base = baselines.get("test", {}) if isinstance(baselines, dict) else {}

        weights_s = summary.get("weights_spearman", {}) if isinstance(summary.get("weights_spearman"), dict) else {}
        policy_s = summary.get("policy_spearman", {}) if isinstance(summary.get("policy_spearman"), dict) else {}

        row: dict[str, object] = {
            "run_dir": str(rd),
            "exp": summary.get("exp"),
            "dataset": summary.get("dataset"),
            "scene": summary.get("scene"),
            "feature": summary.get("feature"),
            "seed": summary.get("seed"),
            "best_weights_val_spearman": summary.get("best_weights_val_spearman"),
            "weights_val_spearman": weights_s.get("val"),
            "best_policy_val_spearman": summary.get("best_policy_val_spearman"),
            "policy_val_spearman": policy_s.get("val"),
            "policy_test_spearman": policy_s.get("test"),
            "baseline_r2_val": val_base.get("r2"),
            "baseline_rmse_val": val_base.get("rmse"),
            "baseline_mae_val": val_base.get("mae"),
            "baseline_nse_val": val_base.get("nse"),
            "baseline_tildeq_val": val_base.get("tildeq"),
            "baseline_prp_rank_val": val_base.get("prp_rank"),
            "baseline_r2_test": test_base.get("r2"),
            "baseline_rmse_test": test_base.get("rmse"),
            "baseline_mae_test": test_base.get("mae"),
            "baseline_nse_test": test_base.get("nse"),
            "baseline_tildeq_test": test_base.get("tildeq"),
            "baseline_prp_rank_test": test_base.get("prp_rank"),
        }
        report_rows.append(row)

    if report_rows:
        from sub_src.io_utils import write_csv, write_json

        csv_path = out_base / f"run_all_report_{timestamp}_seed{args.seed}.csv"
        json_path = out_base / f"run_all_report_{timestamp}_seed{args.seed}.json"
        fields = list(report_rows[0].keys())
        write_csv(csv_path, fields, report_rows)
        write_json(json_path, {"runs": report_rows})
        print(str(csv_path))


if __name__ == "__main__":
    main()
