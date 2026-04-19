# Experiment Layout

The experiment tree is organized by experiment tier first, then method, model,
task, precision, and run family:

```text
experiments/
  main/
    <method>/
      <model>/
        <task>/
          <precision>/
            <run_family>/
  pilot/
    <method>/
      <model>/
        <task>/
          <precision>/
            <run_family>/
  smoke/
    <method>/
      <model>/
        <task>/
          <precision>/
            <run_family>/
```

Method directory names:

- `mezo`
- `lozo`
- `hizoo`
- `sparse_mezo`

Tier meaning:

- `main`: formal long-running sweeps and main experiment outputs
- `pilot`: pilot sweeps, speed benchmarks, and smaller diagnostic runs
- `smoke`: smoke tests, validation runs, and one-off quick checks

Notes:

- `_shared/` stores shared sweep launchers, summaries, and analysis helpers for a
  tier.
- `archive_runs/` stores preserved historical result payloads when a rerun or
  recovery was needed; failed or unfinished logs are removed during cleanup.
- `logs/` should only contain logs for completed runs that are still worth
  keeping.
