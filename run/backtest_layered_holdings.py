# backtest_layered_holdings.py - true rolling sleeve holdings runner
import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.runtime import build_v9_backtest_config, load_backtest_runtime, load_dl_predictor
from backtest.runners import LayeredBacktestParams, run_layered_backtest_once


def parse_args():
    parser = argparse.ArgumentParser(description="True rolling sleeve holdings backtest")
    parser.add_argument("--checkpoint", default="checkpoints/ultimate_v7_gat_best.pt")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--portfolio-mode",
        choices=["optimizer", "optimizer_projected", "optimizer_mvo", "simple_ls", "simple_long"],
        default="optimizer_projected",
    )
    parser.add_argument("--holding-days", type=int, default=5)
    parser.add_argument("--rebalance-freq", type=int, default=1)
    parser.add_argument("--top-frac", type=float, default=0.10)
    parser.add_argument("--adv-mode", choices=["execution", "weight_cap", "both"], default="execution")
    parser.add_argument("--layer-scale", default="auto")
    parser.add_argument("--output-dir", default="backtest_results_layered")
    parser.add_argument("--optimizer-base-mode", choices=["simple_ls", "simple_long"], default="simple_ls")
    parser.add_argument("--optimizer-exposure-control", choices=["none", "beta"], default="beta")
    parser.add_argument("--optimizer-beta-limit", type=float, default=0.05)
    parser.add_argument("--optimizer-dollar-neutral", dest="optimizer_dollar_neutral", action="store_true", default=True)
    parser.add_argument("--no-optimizer-dollar-neutral", dest="optimizer_dollar_neutral", action="store_false")
    parser.add_argument("--mvo-risk-aversion", type=float, default=1.0)
    parser.add_argument("--mvo-lr", type=float, default=0.02)
    parser.add_argument("--mvo-n-iter", type=int, default=200)
    return parser.parse_args()


def load_data_and_predictor(args):
    cfg = build_v9_backtest_config(target_horizon=args.holding_days)

    print("构建截面数据集...")
    runtime = load_backtest_runtime(cfg, use_cache=True)
    print(f"股票数 {len(runtime.price_dict)}")

    predictor = load_dl_predictor(args.checkpoint, runtime.train, runtime.cfg, args.device)
    return runtime.cfg, predictor, runtime.val, runtime.price_dict, runtime.vol_dict


def main():
    os.chdir(PROJECT_ROOT)
    args = parse_args()
    cfg, predictor, val, price_dict, vol_dict = load_data_and_predictor(args)

    params = LayeredBacktestParams(
        holding_days=args.holding_days,
        rebalance_freq=args.rebalance_freq,
        adv_mode=args.adv_mode,
        portfolio_mode=args.portfolio_mode,
        top_frac=args.top_frac,
        optimizer_base_mode=args.optimizer_base_mode,
        optimizer_exposure_control=args.optimizer_exposure_control,
        optimizer_beta_limit=args.optimizer_beta_limit,
        optimizer_dollar_neutral=args.optimizer_dollar_neutral,
        mvo_risk_aversion=args.mvo_risk_aversion,
        mvo_lr=args.mvo_lr,
        mvo_n_iter=args.mvo_n_iter,
        layer_scale=args.layer_scale,
    )
    mode_name = f"{args.portfolio_mode}_layered_h{args.holding_days}_r{args.rebalance_freq}"
    run_layered_backtest_once(
        predictor,
        val,
        price_dict,
        vol_dict,
        cfg,
        params,
        label=mode_name,
        output_dir=args.output_dir,
        print_extended=True,
    )


if __name__ == "__main__":
    main()
