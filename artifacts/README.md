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
It is excluded by `.gitignore` and is available as a
[private GitHub Release asset](https://github.com/ArmstrongInCN/GIFT/releases/tag/reference-checkpoints-20260913).
A plain clone does not include this file. Use an authenticated GitHub CLI account
with access to the private repository, then run from the repository root:

```bash
gh release download reference-checkpoints-20260913 --repo ArmstrongInCN/GIFT --pattern uno_terminal_epoch150.pt --dir artifacts/formal/uno
python scripts/verify_checkpoints.py
```

The download command refuses to overwrite an existing file; if it is already
present, run the verifier rather than downloading over it. Its expected size is
121,928,439 bytes and SHA-256 is
`578186debbffea63afc0805b1a59db3f8f23e79202c8d272b8c45d21493021b4`.
The browser download URL in the catalog also requires private-repository access;
it is not an anonymous download endpoint. Never put a token in that URL.

U-NO 已作为上述私有 Release 的附件提供，普通 clone 仍不包含它。使用有仓库访问权的
GitHub 账号登录 `gh` 后，在仓库根目录执行上述下载与校验命令；已有文件不会被覆盖。
该附件仍是原实验参照权重，不是通过全量验收的新训练结果。

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
