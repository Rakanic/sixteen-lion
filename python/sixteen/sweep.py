"""Run the cross product of a sweep config, one run per output directory.

    python -m sixteen.sweep configs/milestone.yaml              # everything
    python -m sixteen.sweep configs/milestone.yaml --shard 2/6  # every 6th run, offset 2
    python -m sixteen.sweep configs/milestone.yaml --list

Runs whose metrics.json exists are skipped, so a sweep can be resumed and
sharded across processes. Results land in out/<name>/<run>/.
"""

import argparse
import itertools
import json
import pathlib
import traceback

import yaml

from .train import save_reference_crops, train

ROOT = pathlib.Path(__file__).resolve().parents[2]


def expand(cfg):
    sweep = cfg["sweep"]
    keys = list(sweep)
    for combo in itertools.product(*(sweep[k] for k in keys)):
        run = dict(zip(keys, combo))
        grid_name = run.pop("grids")
        loss_name = run.pop("loss", "texel")
        extra = "".join(f"_{k}-{v}" for k, v in run.items() if k not in ("material", "fmt"))
        name = f"{run['material']}/{grid_name}/{loss_name}/{run['fmt']}{extra}"
        kw = {**cfg.get("train", {}), **cfg.get("losses", {}).get(loss_name, {}), **run}
        kw["grids"] = [tuple(g) for g in cfg["grids"][grid_name]]
        yield name, kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = ROOT / "out" / cfg["name"]
    k, n = map(int, args.shard.split("/"))
    runs = list(expand(cfg))
    if args.list:
        for name, kw in runs:
            print(name, kw)
        print(len(runs), "runs")
        return
    if k == 0:
        for m in cfg["sweep"]["material"]:
            if not (out / m / "reference").exists():
                save_reference_crops(m, out / m / "reference")
    for i, (name, kw) in enumerate(runs):
        if i % n != k or (out / name / "metrics.json").exists():
            continue
        print(f"[{i + 1}/{len(runs)}] {name}", flush=True)
        try:
            r = train(out_dir=out / name, **kw)
            print(
                f"  bpp {r['bpp_total']:.2f}  texel {r['psnr_texel']:.2f} dB  "
                f"shaded {r['psnr_shaded']:.2f} dB  FLIP {r['flip']:.4f}  ({r['train_s']:.0f} s)",
                flush=True,
            )
        except Exception:
            traceback.print_exc()
            (out / name).mkdir(parents=True, exist_ok=True)
            (out / name / "error.txt").write_text(traceback.format_exc())


if __name__ == "__main__":
    main()
