#!/usr/bin/env python3
import argparse
import os
import re
import signal
import subprocess
import sys
import threading
import time


NAN_GUARD_EXIT_CODE = 86
NAN_RE = re.compile(r"(?i)\bnan\b")
RESET_MARKERS = (
    "log probabilities of the option tokens",
    "prediction scores",
    "eval_loss",
    "train_loss",
    "\"loss\":",
    "'loss':",
    "projected_grad",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a command, mirror stdout/stderr to log files, and stop it after too many consecutive NaN detections."
    )
    parser.add_argument("--cwd", default=None, help="Working directory for the child process.")
    parser.add_argument("--stdout-log", required=True, help="Path to the stdout log file.")
    parser.add_argument("--stderr-log", required=True, help="Path to the stderr log file.")
    parser.add_argument(
        "--max-consecutive-nan",
        type=int,
        default=1,
        help="Terminate the child process after this many consecutive NaN tokens are observed. Default=1 means the first non-ignorable NaN will skip the current h.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run after '--'.")
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing command to execute")
    return args


def is_ignorable_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    progress_markers = ("it/s", "%|", "ETA", "████", "▏", "▎", "▍", "▌", "▋", "▊", "▉")
    return any(marker in stripped for marker in progress_markers)


def should_reset_counter(line: str) -> bool:
    lowered = line.strip().lower()
    if not lowered:
        return False
    if is_ignorable_line(line):
        return False
    return any(marker in lowered for marker in RESET_MARKERS)


class GuardState:
    def __init__(self, max_consecutive_nan: int):
        self.max_consecutive_nan = max_consecutive_nan
        self.consecutive_nan_tokens = 0
        self.triggered = False
        self.lock = threading.Lock()

    def note_line(self, line: str) -> int:
        nan_count = len(NAN_RE.findall(line))
        with self.lock:
            if nan_count > 0:
                self.consecutive_nan_tokens += nan_count
            elif should_reset_counter(line):
                self.consecutive_nan_tokens = 0
            if self.consecutive_nan_tokens >= self.max_consecutive_nan:
                self.triggered = True
            return self.consecutive_nan_tokens

    def is_triggered(self) -> bool:
        with self.lock:
            return self.triggered


def terminate_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.2)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def stream_worker(stream, sink, state: GuardState, process: subprocess.Popen, channel_name: str) -> None:
    while True:
        line = stream.readline()
        if line == "":
            break
        sink.write(line)
        sink.flush()
        count = state.note_line(line)
        if state.is_triggered():
            notice = (
                f"[nan-guard] reached {count} consecutive NaN tokens on {channel_name}; "
                f"terminating child process {process.pid}.\n"
            )
            sink.write(notice)
            sink.flush()
            terminate_process_tree(process)
            break


def main() -> int:
    args = parse_args()

    os.makedirs(os.path.dirname(args.stdout_log) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.stderr_log) or ".", exist_ok=True)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    with open(args.stdout_log, "w", encoding="utf-8") as stdout_log, open(
        args.stderr_log, "w", encoding="utf-8"
    ) as stderr_log:
        command_display = " ".join(args.command)
        header = (
            f"[nan-guard] command={command_display}\n"
            f"[nan-guard] cwd={args.cwd or os.getcwd()} max_consecutive_nan={args.max_consecutive_nan}\n"
        )
        stdout_log.write(header)
        stderr_log.write(header)
        stdout_log.flush()
        stderr_log.flush()

        process = subprocess.Popen(
            args.command,
            cwd=args.cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            preexec_fn=os.setsid,
        )

        state = GuardState(args.max_consecutive_nan)

        stdout_thread = threading.Thread(
            target=stream_worker,
            args=(process.stdout, stdout_log, state, process, "stdout"),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=stream_worker,
            args=(process.stderr, stderr_log, state, process, "stderr"),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        return_code = process.wait()
        stdout_thread.join()
        stderr_thread.join()

        if state.is_triggered():
            footer = (
                f"[nan-guard] terminated child after {state.consecutive_nan_tokens} consecutive NaN tokens.\n"
            )
            stdout_log.write(footer)
            stderr_log.write(footer)
            stdout_log.flush()
            stderr_log.flush()
            return NAN_GUARD_EXIT_CODE

        footer = f"[nan-guard] child exited with code {return_code}.\n"
        stdout_log.write(footer)
        stderr_log.write(footer)
        stdout_log.flush()
        stderr_log.flush()
        return int(return_code)


if __name__ == "__main__":
    raise SystemExit(main())
