# Final Revision Audit

[x] Panel C 不再回归理论 h_ref
[x] Panel C 使用 empirical rho-window center / empirical optimum
[x] empirical accuracy interval 明确为 best-1 percentage point
[x] INT4 明确标记 no tau=1 certified window
[x] FP32/FP16 标为 empirical-only
[x] BF16 无数据时已删除
[x] RoBERTa 主表无空 policy
[x] RoBERTa 主表无重复 row
[x] 主表不混合 full/medium
[x] recommended row 不是按 accuracy cherry-pick
[x] OPT 保留 TREC failure
[x] OPT claim 仅为 sanity check
[x] frozen h_ref 与 legacy hstar 已分开
[x] cost 表无无法解释的空列
[x] 每个结果都有 source_path
[x] 未新增理论、selector 或拟合模型
[x] 未启动新的大规模训练
