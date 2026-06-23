# Fixed h2_L=1e-3 lowbit-L h-star check

Formula: h*=(Delta^2 G^2 / (16 L^2 d(d+2)))^(1/4). G is the precision-aware low-bit G already used in the corresponding summaries.

| model | task | cleanL h* | auto lowbitL h* (h2) | fixed lowbitL@1e-3 h* | nearest | lowbitL@3e-3 h* |
|---|---:|---:|---:|---:|---:|---:|
| OPT-1.3B | sst-2 | 0.000414078 | 0.00713215 (0.03) | 0.000534396 | 0.0005 | 0.0010728 |
| OPT-1.3B | sst-5 | 0.000387799 | 0.000620109 (0.001) | 0.000620109 | 0.0007 | 0.000973019 |
| RoBERTa-large | sst-2 | 0.00136851 | 0.000621479 (0.001) | 0.000621479 | 0.0007 | 0.000833156 |
| RoBERTa-large | sst-5 | 0.00149894 | 0.000614543 (0.001) | 0.000614543 | 0.0007 | 0.00138603 |
| RoBERTa-large | rte | 0.00133319 | 9.13125e-05 (3e-05) | 0.000543394 | 0.0005 | 0.000849239 |
| RoBERTa-large | mnli | 0.000859823 | 5.09807e-05 (3e-05) | 0.000387847 | 0.0005 | 0.000854447 |
| RoBERTa-large | trec | 0.0017888 | 0.00162425 (0.002) | 0.000585648 | 0.0005 | 0.00163487 |
