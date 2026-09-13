# Working on GIFT / 操作约定

关键操作：每个已支持的模型/实验独立执行，失败不连带重跑已完成任务；从零训练不读已发布权重，续算只接本次完整状态。PINN known/KC 可独立执行但完整预算数值验收尚未通过；open 训练的未验门禁不得绕过，已训练终态快读不解除该门禁。原实验和参照值不可改动，数值先落盘再绘图，输出只写新目录；发布前检查 Git 文件清单，未经所有者明确授权不公开。

- Treat the experimental record `EXPERIMENTS.md` and published reference tables as immutable evidence. Do not change tolerances, remove outliers, or replace reference values to make a run pass.
- 数据通过 `GIFT_DATA_ROOT` 指向仓库外 Zenodo 数据包；用只读模式打开。不要将数据、外部源码、凭据、缓存、个人路径或训练日志提交到 Git。
- Third-party implementations are supplied by their authors' repositories, pinned in `external_sources.json`. Never copy upstream model definitions, optimizers, training loops, or large patches into this repository, even under a new filename.
- Each supported model and experiment has its own command. PINN known/KC execution is enabled but full-budget numerical acceptance has not passed; open training remains gated, independently of trained-state quick readout. Do not restart completed models to recover an unrelated failed experiment. Fresh training must not load published weights; resume may load only the same run's saved model, optimizer, scheduler and random-number states.
- Keep the original numerical profile: seed alone does not fix floating-point behavior. Do not introduce a different interpolation backward pass or global TF32 override without a separately reported validation.
- PINN's public device selector belongs to `training.train_pinn --device cpu|gpu` (CPU default, GPU known/KC only); its NAdam-only primitive CLI remains CPU. Use only a child-process PATH for the independent legacy environment's DLLs. Three bounded GPU checks are not full training acceptance. Preserve the paused old CPU run's frozen source/journal/launch; its resume requires that original source and CPU profile, never the new source or a cross-device continuation. 旧CPU任务须用自身冻结源码与环境续算，不把GPU小门禁或restore-only称为完整M1验收。
- Save numerical results before optional plotting. An interrupted figure renderer must not discard completed predictions or metric tables.
- Use explicit new output directories and fail on accidental overwrite. **禁止直接运行批量删除命令；需要批量删除时必须向用户申请。**
- Do not access or modify an author's external authoritative copy (such as `1_GIFT`) as part of normal project execution. Exclude `backup` and manuscript-writing folders.
- Distinguish cached-table inspection, checkpoint-based inference, and fresh training in reports. A small smoke test or interrupted/resumed toy training run is not full-budget reproduction evidence.
- Before upload, run the repository audit and inspect the exact Git file list. Keep this repository private until its owner explicitly authorizes publication; never purchase storage or enable paid services automatically.
- Record unresolved technical or provenance issues in [verification status](docs/REPRODUCIBILITY_STATUS.md). Explain a genuine blocker promptly instead of silently changing the experiment.
