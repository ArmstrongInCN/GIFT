# Full paired-noise generation / 完整配对噪声生成

**PASS for this specified-input derivation, not complete-project acceptance.**
On 2026-09-13, the independent noise entry point generated both complete M1
observation files from the released clean N64 input. No reference noisy field
was used by generation. A separate read-only verifier compared all four
condition/group arrays: **287,293,440 float32 values, zero bit mismatches and
zero nonfinite values**. Independent seed-zero formula replay also matched
every generated value bitwise; no numeric tolerance was introduced.

**本次指定输入派生验证通过，不等于整个项目验收通过。** 独立入口从已发布的
clean 输入完整生成 1% 和 10% 噪声观测，生成过程中不读取参照噪声。
另行只读验证全部四个条件/分组数组，共 **287,293,440 个 float32 值**，
与参照逐位一致，无非有限值；独立重放随机公式也逐位一致，未设置宽松容差。

## Coverage and own-attempt continuation / 覆盖与自身续算

| Array in each noise condition | Shape | Values per condition |
| --- | --- | ---: |
| Training, IDs 0–49 | `[50,501,64,64]` | 102,604,800 |
| Validation, IDs 50–69 | `[20,501,64,64]` | 41,041,920 |

Both conditions retain float32 vorticity, float64 time and int64 trajectory
IDs. All time/ID coordinates and their stored bits matched. The time axis is
0 through 10 in 0.02 increments; spatial semantics are `[y,x]` from the
clean-input contract, not HDF5 dimension labels (which are empty).

The fresh first process paused after chunk **1,601 of 2,240**, taking
**9.081248599977698 s**. A separate process resumed only that attempt's saved
state and completed the remaining **639 chunks in 4.443982800003141 s**.
The attempt UUID is `ec98b2e1-676c-4178-9eaf-55897c1e739e`.
The independent verifier then took **16.56199999997625 s**; these are three
separate measured durations, not one uninterrupted process.

两次生成进程分别暂停、续算同一任务，未换用参照文件。暂停点与终点的完整
MT19937 状态（624 个状态字、位置、高斯缓存标志及缓存值的 float64 位模式）
均与独立连续重放一致。检查点保留，未删除或替换。

Whole-group clean statistics were independently recomputed in the original
trajectory-reduction order. Count, mean, population standard deviation and
sum of squares matched the stored run statistics exactly. The reference H5
groups have no authoritative clean-statistic attributes; no nonexistent
attribute comparison is claimed. Source, input, generated-output, checkpoint
and execution-receipt hashes were unchanged across the verifier.

## Evidence and limits / 证据与限制

[result.json](result.json) records all comparisons, runtime versions, checkpoint
boundaries, metadata differences and file hashes. [manifest.json](manifest.json)
binds the two portable files. The original local verification result SHA256 is
`BDB35517E2AE2A7A452475FCD8146ECB22A1639BEE4312E8B386C512F1491E2B`.
Raw logs, generated H5 files, checkpoint files, personal paths and input data
are not distributed in this evidence folder.

- **No clean reintegration:** the released clean input was supplied. This does
  not reproduce its Navier–Stokes integration or validate all clean datasets.
- **Not whole-file equality:** new H5 metadata and whole-file hashes differ
  from the historical containers despite exact array contents. Legacy absolute
  provenance paths were not copied into generated files.
- **No full regenerated collection:** assembly requires its own freshly
  generated clean parent. These released-clean-derived outputs cannot replace
  that missing parent or bypass input-hash/provenance gates.
- **No training acceptance:** no model training, PyTorch or TensorFlow imports
  occurred in the verifier. Existing U-NO, S2 and PINN numerical failures are
  unchanged. This one fixed-runtime test is not a cross-device guarantee.

本记录不声称重新积分 clean、不声称 H5 整文件哈希相同、不声称已组装完整
重生成数据集合，也不改变已有模型的数值失败结论。原发布输入保持不变。
See [data-generation instructions](../../docs/DATA_GENERATION.md) and the
[overall verification status](../../docs/REPRODUCIBILITY_STATUS.md).
