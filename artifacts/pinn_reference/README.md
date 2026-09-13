# Recovered PINN reference state / 恢复的 PINN 参照状态

These six numeric-only NPZ archives preserve the original terminal checkpoint
arrays: 82 float32 tensors each, including 18 network arrays, coefficients,
mask and historical optimizer/accumulator state. They contain no `.meta` graph,
model class, upstream source or pickle object. See [manifest.json](manifest.json)
for each archive/array SHA256, original-file identities and the key/name map.

六套文件对应三个噪声条件的 PINN-SR（`open`）与 PINN-SR-KC（`known`）。
它们是 **reference_not_resume**：保存了原终点数值，不是本候选从零训练产物，
也不是新训练器可直接恢复的 journal。optimizer 槽存在不等于续算已验证。

From the clone root, with NumPy installed:

```bash
python -B -m adapters.pinn_reference --condition noise_001 --mode known
```

在仓库根运行上述命令；可分别选择 `noise_000`、`noise_001`、`noise_010` 和
`open` / `known`。读取器核验固定 manifest、整个 NPZ 及全部 82 个数组，再从
实际 `coefficients * coefficient_mask` 按原库项顺序提取 nu、beta、gamma。
它不信任预写参数表，不导入 TensorFlow/Torch，不训练、不前向、不自动下载。

The recovered original method-record chain matches the formal source-evidence
hashes, and all six checkpoint coefficient/mask vectors match the native and
enriched results exactly. Original endpoint file hashes were observed during
recovery; no historical endpoint hash seal was found. The manifest preserves
that distinction. All 90 open-library coefficients are nonzero: correct-term
readout is not a claim of recovered sparse structure.

2026-09-10 实际在 Python 3.8 / NumPy 1.21.6 和 Python 3.10 / NumPy 2.2.6 下
各通过 7 项纯 NumPy 读取/完整性测试；包含六套实际参数读出、无框架导入、
篡改/错误 schema/非数值对象拒绝，以及不信任伪造 parameters 字段的检查。
这些测试不证明终点网络可在新环境等价推理，也不替代完整 PINN 训练验收。
使用范围见 [M1 说明](../../docs/adapters/M1.md)。
