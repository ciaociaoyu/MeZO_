#!/bin/bash

# "8h" here means 8 candidate h values, not 8 hours of runtime.
# This grid is a reduced subset of the canonical 14-value grid, keeping the
# mid-range values that were stable/useful in prior pilot runs.
H_VALUES=(
  1e-6
  3e-6
  1e-5
  3e-5
  1e-4
  3e-4
  1e-3
  3e-3
)
