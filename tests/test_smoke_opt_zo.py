import csv
import json
import subprocess
import sys
from pathlib import Path


def test_smoke_opt_zo_local_optconfig(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "smoke_opt_zo"
    cmd = [
        sys.executable,
        str(repo / "tools" / "smoke_opt_zo.py"),
        "--model_id",
        "local-opt-tiny",
        "--methods",
        "env,dense,fake_int8,sparse,residual,checkpoint",
        "--device",
        "cpu",
        "--batch_size",
        "1",
        "--max_seq_len",
        "16",
        "--k_dirs",
        "1",
        "--h_grid",
        "3e-3",
        "--sparse_p",
        "0.1",
        "--sparse_h_active",
        "6e-3",
        "--max_touched_params",
        "1",
        "--output_dir",
        str(out_dir),
    ]
    result = subprocess.run(cmd, cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    env = json.loads((out_dir / "env.json").read_text())
    assert env["env"]["torch_version"]

    rows = [json.loads(line) for line in (out_dir / "smoke_results.jsonl").read_text().splitlines()]
    assert rows
    assert not [row for row in rows if row["status"] == "fail"]
    passed_methods = {row["method"] for row in rows if row["status"] == "pass"}
    for method in {"env", "dense", "fake_int8", "sparse", "residual", "checkpoint"}:
        assert method in passed_methods

    with (out_dir / "smoke_summary.csv").open(newline="") as f:
        summary_rows = list(csv.DictReader(f))
    assert len(summary_rows) == len(rows)
