# Working on GIFT / 项目操作约定

- Only edit this project and explicitly selected run directories. External datasets and upstream checkouts are read-only.
- Keep data outside the repository. Use `GIFT_DATA_ROOT` and `GIFT_EXTERNAL_ROOT`; never search for another author's working copy as a fallback.
- 外部算法只允许必要、少量且能逐项用简单语言说明的适配。不得接管自动微分、替换反向算子或重新实现优化器内核来追求特定结果。
- Record every external-source adjustment in `docs/EXTERNAL_ADAPTATIONS.md`, separately from experimental settings. Use fixed upstream commits and verify source hashes before loading code.
- Train one model per command. Resume only that run's committed model, optimizer, scheduler and random state. State the checkpoint boundary explicitly; restarting an uncommitted phase is not an internal-optimizer-state resume.
- Every GIFT trainer uses `--execution auto`: CUDA graphs for fresh GPU training, eager on CPU, and the recorded backend for ordinary resume. `--execution eager` keeps the reference training path. Prediction remains eager unless explicitly requested via `--gift-execution cuda-graph`. Keep scientific budgets, precision and pre-update safety checks. Never bypass source/runtime/backend checkpoint identity checks. See docs/PERFORMANCE.md; benchmark updates are disposable, not scientific training progress.
- High-frequency training layouts are segment-local differentiable views, never persistent learned-weight caches. Only verified full batches (16 derivative / 5 rollout trajectories) use them; other shapes keep the reference graph. Preserve parameter storage, native autograd, default prediction, and refresh/release the views at every backward boundary.
- Define data access, training budget, model selection and evaluation rules before training. Equal epochs, equal parameter updates and equal compute are different comparisons; never label one as another.
- Prediction models use the common 1,000-trajectory training set. Full-data GIFT trains its generator and branch for 500 trajectory epochs EACH; report 500 + 500, never 500 total. Each baseline uses 500 epochs. Preserve completed 50-trajectory prediction models as GIFT-Lite; Lite describes reduced training data, not a smaller architecture. M1 training and results remain unchanged. See docs/TRAINING_PROTOCOL.md.
- Do not use test outcomes to tune settings, omit difficult trajectories or choose checkpoints. Save numerical outputs before plotting.
- Keep development diaries, intermediate implementation variants and comparisons to other project folders outside the delivered project. Current experiment settings, source identities, result provenance and known limitations must remain accurate.
- Only update `EXPERIMENTS.md` from completed, source-bound measurements. Make the smallest necessary changes to methods, tables, figures and conclusions; do not present a pending computation as a measured result.
- All experiment-document images are SVG. Preserve the existing visual design; regenerate arrays from the selected model runs and figures from their numerical outputs. Check table/figure provenance together. Never enforce a preferred method ranking in plotting code.
- GitHub visibility must remain private until explicitly authorized otherwise. Do not rewrite remote history or enable paid services automatically.
- 禁止直接运行批量删除命令；任何批量删除操作必须先向用户申请。不得覆盖现有训练或实验输出。
- 跨学科工作只能通过新增入口进行，不得改动既有模块：`scripts/generate_crossdomain_data.py` 生成 `fixture_only` 诊断观测，`scripts/run_crossdomain.py` 以项目自带的 `PredictionTrainingConfig(fixture_only=True)` 训练生成元并用 `gift.identified` 做递归评估。fixture 产物不得表述为正式实验结果，也不得与已发布表格混排；引用其数值时须与对应 `summary/summary.json` 的模型与数据哈希绑定。
- 跨学科运行同样遵守既有约定：每次一个新输出目录、默认只读计划、`--execute` 才落盘、输出位于仓库与 `GIFT_DATA_ROOT` 之外；一个方程对应一个固定参数集（单一自治系统），参数随轨迹变化、二阶系统或非周期区域都不在已验证范围内。
