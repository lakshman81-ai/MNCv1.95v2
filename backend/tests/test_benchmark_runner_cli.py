import json
import subprocess
import sys
from pathlib import Path


def test_benchmark_runner_l3_cli(tmp_path: Path):
    output_dir = tmp_path / "benchmarks"
    cmd = [
        sys.executable,
        "-m",
        "backend.benchmarks.benchmark_runner",
        "--level",
        "L3",
        "--output",
        str(output_dir),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    summary_path = output_dir / "summary.csv"
    leaderboard_path = output_dir / "leaderboard.json"

    assert summary_path.exists(), "Summary CSV not generated"
    assert leaderboard_path.exists(), "Leaderboard not generated"

    summary_content = summary_path.read_text()
    assert "L3" in summary_content, "L3 row missing from summary"

    metrics_files = list(output_dir.glob("L3_*_metrics.json"))
    assert metrics_files, "L3 metrics were not saved"

    metrics = json.loads(metrics_files[0].read_text())
    assert metrics.get("level") == "L3"

    leaderboard = json.loads(leaderboard_path.read_text())
    assert metrics.get("name") in leaderboard
