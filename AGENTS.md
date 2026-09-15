# Working on GIFT / 项目操作约定

- Only edit this project and explicitly selected run directories. External datasets and upstream checkouts are read-only.
- Keep data outside the repository. Use `GIFT_DATA_ROOT` and `GIFT_EXTERNAL_ROOT`; never search for another author's working copy as a fallback.
- 外部算法只允许必要、少量且能逐项用简单语言说明的适配。不得接管自动微分、替换反向算子或重新实现优化器内核来追求特定结果。
- Record every external-source adjustment in `docs/EXTERNAL_ADAPTATIONS.md`, separately from experimental settings. Use fixed upstream commits and verify source hashes before loading code.
- Train one model per command. Resume only that run's committed model, optimizer, scheduler and random state. State the checkpoint boundary explicitly; restarting an uncommitted phase is not an internal-optimizer-state resume.
- GIFT acceleration is opt-in: `--execution cuda-graph` for full-data training, `--gift-execution cuda-graph` for prediction experiments. Keep eager reference behavior, scientific budgets, precision and pre-update safety checks. Never bypass source/runtime/backend checkpoint identity checks. See docs/PERFORMANCE.md; benchmark updates are disposable, not scientific training progress.
- Define data access, training budget, model selection and evaluation rules before training. Equal epochs, equal parameter updates and equal compute are different comparisons; never label one as another.
- Prediction models use the common 1,000-trajectory training set. Full-data GIFT trains its generator and branch for 500 trajectory epochs EACH; report 500 + 500, never 500 total. Each baseline uses 500 epochs. Preserve completed 50-trajectory prediction models as GIFT-Lite; Lite describes reduced training data, not a smaller architecture. M1 training and results remain unchanged. See docs/TRAINING_PROTOCOL.md.
- Do not use test outcomes to tune settings, omit difficult trajectories or choose checkpoints. Save numerical outputs before plotting.
- Keep development diaries, intermediate implementation variants and comparisons to other project folders outside the delivered project. Current experiment settings, source identities, result provenance and known limitations must remain accurate.
- Only update `EXPERIMENTS.md` from completed, source-bound measurements. Make the smallest necessary changes to methods, tables, figures and conclusions; do not present a pending computation as a measured result.
- All experiment-document images are SVG. Preserve the existing visual design; regenerate arrays from the selected model runs and figures from their numerical outputs. Check table/figure provenance together. Never enforce a preferred method ranking in plotting code.
- GitHub visibility must remain private until explicitly authorized otherwise. Do not rewrite remote history or enable paid services automatically.
- 禁止直接运行批量删除命令；任何批量删除操作必须先向用户申请。不得覆盖现有训练或实验输出。
