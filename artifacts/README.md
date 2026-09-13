# Trained reference models / 已训练参照模型

See [CHECKPOINTS.json](CHECKPOINTS.json) for exact sizes, SHA256 hashes and
distribution status. The reference `.pt` files are original experimental weights,
preserved without alteration. They are inference artifacts, not resume journals.
The separate [PINN numeric references](pinn_reference/README.md) are safe NPZ
exports of recovered terminal arrays, not byte-identical copies of the original
TensorFlow checkpoint files. Their source hashes and conversion checks are
recorded; their role is `reference_not_resume`, even though numeric optimizer
arrays are retained. 原 `.pt` 权重与 PINN 安全数值转换分别标注来源，均不能冒充本次训练断点。

U-NO's 121,928,439-byte terminal file is larger than GitHub's ordinary Git limit.
It is excluded by `.gitignore` and is intended for a **private GitHub release
asset**. Until the catalog contains a verified download location, a clean clone
does not include this file. The local candidate has a byte-exact copy for tests.

From the repository root, run `python scripts/verify_checkpoints.py` after
obtaining the listed weights. A missing or corrupt file fails explicitly. This
command verifies bytes only; it does not run a model or prove retraining.

根目录及子目录原有的 `manifest.json`、训练报告和训练曲线均是历史来源记录，
并非此次候选的训练验收结果；其中的旧项目路径不要求在读者电脑上存在。
当前分发以 `CHECKPOINTS.json` 为准，数据和 PINN 采样文件位于独立的数据包。

独立训练器输出 `model.pt` 或相应 GIFT `.pt`；训练过程的完整状态保存在它自身的
checkpoint journal 中。请勿将这里的终点权重当作“从零训练”的初始化，或将其
改名后放入 journal 冒充断点。

The new candidate has not yet demonstrated full-budget reproduction of every
reference checkpoint. See [the reproduction status](../docs/REPRODUCIBILITY_STATUS.md).
