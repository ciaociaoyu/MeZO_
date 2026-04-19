#!/bin/bash

hsweep_run_completed() {
  local run_summary_path="$1"
  local summary_file="$2"
  local h_value="$3"

  python - "${run_summary_path}" "${summary_file}" "${h_value}" <<'PY'
import json
import os
import sys

run_summary_path, summary_file, h_value = sys.argv[1:]

if run_summary_path and os.path.exists(run_summary_path):
    try:
        with open(run_summary_path, "r", encoding="utf-8") as f:
            json.load(f)
        sys.exit(0)
    except Exception:
        pass

if summary_file and os.path.exists(summary_file):
    try:
        with open(summary_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                if str(record.get("h")) == h_value and record.get("status") == "completed":
                    sys.exit(0)
    except Exception:
        pass

sys.exit(1)
PY
}

hsweep_drop_h_rows() {
  local jsonl_path="$1"
  local h_value="$2"

  if [[ -z "${jsonl_path}" || ! -f "${jsonl_path}" ]]; then
    return 0
  fi

  python - "${jsonl_path}" "${h_value}" <<'PY'
import json
import os
import sys
from tempfile import NamedTemporaryFile

jsonl_path, h_value = sys.argv[1:]
parent = os.path.dirname(jsonl_path) or "."

with open(jsonl_path, "r", encoding="utf-8") as src, NamedTemporaryFile(
    "w", delete=False, dir=parent, encoding="utf-8"
) as dst:
    tmp_path = dst.name
    for raw_line in src:
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except Exception:
            dst.write(raw_line)
            continue
        if str(record.get("h")) == h_value:
            continue
        dst.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

os.replace(tmp_path, jsonl_path)
PY
}

hsweep_cleanup_paths() {
  local path
  for path in "$@"; do
    if [[ -n "${path}" && -e "${path}" ]]; then
      rm -rf "${path}"
    fi
  done
}
