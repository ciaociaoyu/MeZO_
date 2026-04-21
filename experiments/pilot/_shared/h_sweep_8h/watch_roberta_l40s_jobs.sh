#!/bin/bash

set -euo pipefail

LOG_DIR="/scratch/jy03364/MeZO_/experiments/pilot/_shared/h_sweep_8h/logs"
mkdir -p "${LOG_DIR}"

JOB_IDS=(
  44586568
  44586569
  44586570
  44586571
  44586572
  44586573
)

STATUS_LOG="${LOG_DIR}/watch_roberta_l40s_jobs.log"
STATE_FILE="${LOG_DIR}/watch_roberta_l40s_jobs.state"

touch "${STATUS_LOG}"

declare -A PREV_STATE=()
if [[ -f "${STATE_FILE}" ]]; then
  while IFS='|' read -r job_id state; do
    [[ -n "${job_id}" ]] || continue
    PREV_STATE["${job_id}"]="${state}"
  done < "${STATE_FILE}"
fi

while true; do
  now="$(date --iso-8601=seconds)"
  tmp_state="$(mktemp)"
  any_running=0
  any_present=0

  for job_id in "${JOB_IDS[@]}"; do
    line="$(squeue -h -j "${job_id}" -o '%i|%T|%R|%M' || true)"
    if [[ -n "${line}" ]]; then
      any_present=1
      IFS='|' read -r seen_id state reason runtime <<< "${line}"
      prev="${PREV_STATE[${job_id}]:-}"
      echo "${job_id}|${state}" >> "${tmp_state}"
      if [[ "${state}" == "RUNNING" ]]; then
        any_running=1
      fi
      if [[ "${state}" != "${prev}" ]]; then
        printf '[%s] job=%s state=%s runtime=%s reason=%s\n' "${now}" "${job_id}" "${state}" "${runtime}" "${reason}" >> "${STATUS_LOG}"
        PREV_STATE["${job_id}"]="${state}"
      fi
      continue
    fi

    sacct_line="$(sacct -n -X -j "${job_id}" --format=JobIDRaw,State,Elapsed,ExitCode | awk -F'|' -v id="${job_id}" '$1 == id {print $0; exit}' || true)"
    if [[ -n "${sacct_line}" ]]; then
      any_present=1
      IFS='|' read -r seen_id state elapsed exit_code <<< "${sacct_line}"
      prev="${PREV_STATE[${job_id}]:-}"
      echo "${job_id}|${state}" >> "${tmp_state}"
      if [[ "${state}" != "${prev}" ]]; then
        printf '[%s] job=%s state=%s elapsed=%s exit=%s\n' "${now}" "${job_id}" "${state}" "${elapsed}" "${exit_code}" >> "${STATUS_LOG}"
        PREV_STATE["${job_id}"]="${state}"
      fi
    fi
  done

  mv "${tmp_state}" "${STATE_FILE}"

  if [[ "${any_present}" -eq 0 ]]; then
    printf '[%s] watcher exiting: no tracked jobs found in squeue or sacct\n' "${now}" >> "${STATUS_LOG}"
    break
  fi

  if [[ "${any_running}" -eq 1 ]]; then
    sleep 60
  else
    sleep 120
  fi
done
