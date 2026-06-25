# RoBERTa INT4 experiment takeaways

## 中文

1. 旧公式不需要完全废弃；它可以作为 coarse-envelope 的 h_env。
2. h_sharp 是更细的 interval-aware probe 诊断，不应声称总能提升训练 accuracy。
3. practical 主方法更适合用 h_safe：default 在 sharp window 内就保留 1e-3，否则才覆盖。
4. 已有 full 训练会被复用；缺失的 h_sharp/full 训练在 manifest 中列出，没有编造。
5. 如果 h_env 落在 sharp window 内，旧实验可以保留并解释为 coarse-envelope radius。

## English

1. The old formula should not be discarded; it remains a coarse-envelope radius.
2. h_sharp is a sharper interval-aware perturbation diagnostic, not a guarantee of better accuracy.
3. h_safe is the practical rule: keep default when it is inside the sharp window; override only when unsafe.
4. Existing full training results are reused; missing sharp/full rows are explicitly reported.
5. If h_env lies inside the sharp window, previous experiments remain valid as coarse-envelope runs.
