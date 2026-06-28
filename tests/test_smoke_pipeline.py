from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from sublevel_detect import paths


def test_smoke_with_ablation_writes_main_and_ablation_outputs() -> None:
    output_root = Path(tempfile.mkdtemp(prefix="sublevel_detect_smoke_", dir=paths.PROJECT_ROOT))
    try:
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        completed = subprocess.run(
            [
                sys.executable,
                str(paths.PROJECT_ROOT / "run.py"),
                "--mode",
                "smoke",
                "--output",
                str(output_root),
                "--exclude",
                "hpopt",
                "--ablation",
            ],
            cwd=str(paths.PROJECT_ROOT),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=240,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert (output_root / "main" / "fullscan" / "decision.json").exists()
        assert (output_root / "main" / "fullscan" / "scan_summary.csv").exists()
        assert (output_root / "main" / "paper_summary.json").exists()
        assert (output_root / "main" / "paper_summary.md").exists()
        assert (output_root / "ablation" / "ablation_summary.csv").exists()
        assert (output_root / "ablation" / "ablation_report.md").exists()
    finally:
        shutil.rmtree(output_root, ignore_errors=True)
