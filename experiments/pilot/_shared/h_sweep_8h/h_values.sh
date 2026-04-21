#!/bin/bash

# "8h" here means 8 candidate h values, not 8 hours of runtime.
# Canonical pilot grid as of 2026-04-20: 8 decade-spaced values from 1e-2 down
# to 1e-9. Older mid-range 3e-* grids are retained only in historical artifacts.
H_VALUES=(
  1e-2
  1e-3
  1e-4
  1e-5
  1e-6
  1e-7
  1e-8
  1e-9
)
