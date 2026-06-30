from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def load_status(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def collect_status_rows(root: str | Path) -> List[Dict[str, Any]]:
    root = Path(root)
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.rglob("status.json")):
        try:
            payload = load_status(path)
        except Exception:  # noqa: BLE001
            continue
        payload["status_path"] = str(path)
        rows.append(payload)
    return rows


def format_status_snapshot(rows: Sequence[Dict[str, Any]], now: float | None = None) -> str:
    now = float(time.time() if now is None else now)
    if not rows:
        return f"[{time.strftime('%H:%M:%S')}] no active status files"
    lines = [f"[{time.strftime('%H:%M:%S')}] active jobs={len(rows)}"]
    for row in sorted(rows, key=lambda r: (int(r.get("n_levels", 0)), int(r.get("seed", 0)))):
        age = now - float(row.get("heartbeat", now))
        status = str(row.get("status", "unknown"))
        stage = str(row.get("hyperopt_stage") or row.get("stage") or "unknown")
        lines.append(
            "L{level:02d} seed={seed:03d} {status} {stage} epoch={epoch}/{epochs} "
            "loss={loss:.6g} structure={structure:.6g} vr_bias={vr_bias:.6g} "
            "vr_amp={vr_amp:.6g} valley={valley:.6g} best={best} age={age:.0f}s".format(
                level=int(row.get("n_levels", 0)),
                seed=int(row.get("seed", 0)),
                status=status,
                stage=stage,
                epoch=int(row.get("epoch", 0)),
                epochs=int(row.get("epochs", 0)),
                loss=safe_float(row.get("loss_total")),
                structure=safe_float(row.get("structure_monitor")),
                vr_bias=safe_float(row.get("loss_vr_late_bias")),
                vr_amp=safe_float(row.get("loss_vr_amplitude_ratio")),
                valley=safe_float(row.get("loss_high_vr_valley_depth")),
                best=int(row.get("best_epoch", 0)),
                age=max(0.0, age),
            )
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor sublevel detection training status files")
    parser.add_argument("--root", default="output/main/fullscan")
    parser.add_argument("--interval", type=int, default=15)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root)
    interval = max(1, int(args.interval))
    while True:
        print(format_status_snapshot(collect_status_rows(root)), flush=True)
        time.sleep(interval)
