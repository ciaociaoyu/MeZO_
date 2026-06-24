# Synthetic Coverage Extension

- Combined raw rows: 6840
- Dimensions: [1000, 10000, 100000, 1000000]
- Delta values: [1e-05, 3e-05, 0.0001, 0.0003, 0.001, 0.003, 0.01]
- Active p values: [0.001, 0.01, 0.05, 0.1, 1.0]
- Group sizes: [64, 128, 256]
- Scale sigmas: [0.0, 0.5, 1.0]

The final synthetic table includes the main sweep plus mid-p/mid-Delta and group-size add-ons. k-direction averaging is saved in `synthetic_k_averaging_window.csv`.
