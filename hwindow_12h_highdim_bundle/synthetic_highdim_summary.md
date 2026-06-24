# Synthetic High-Dimensional Summary

- Rows: 5760
- Dimensions: [1000, 10000, 100000, 1000000]
- Delta values: [1e-05, 0.0001, 0.001, 0.01]
- qbits: [4, 8]
- Active p values: [0.01, 0.1, 1.0]

Main interpretation:
- Larger Delta shifts the visibility boundary to larger h in the interval metrics.
- Larger d increases the random-direction floor used in rho(h), making some h-dependent differences less consequential for convergence.
- Sparse/effective-dimension settings change active fraction and rho windows; this is the high-dimensional mechanism to emphasize.
- The interval-aware crossing metrics are empirical and should be preferred over a single coarse Delta^2/h^2 bound for heterogeneous scales.
