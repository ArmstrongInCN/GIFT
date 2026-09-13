# 新数据的实际实验读取 / Real experimental readers on generated data

**PASS，31.218秒；不是训练或完整实验验收。** 2026-09-13，直接调用现有实验代码的读取函数，使用此前独立核验通过的完整新数据集合。没有mock、改写reader、插值、加载检查点或读取原发布场作为参照。

**PASS in 31.218 s, scoped to actual data readers and cross-file consistency.** Seven top-level calls plus the nested short reader ran on all required 200-trajectory populations. The public `ProjectPaths.from_root` resolver was used with the external collection. Source, manifest and selected-file hashes were checked before/after execution.

| 实际调用 / Actual call | 覆盖 / Coverage |
| --- | --- |
| `load_n64_long` | N64 IDs1000–1199, seven times t=5:0.5:8 |
| `load_fno_test_dt0p02`, N64 M2 schedule | 46-frame history and seven report frames, exact standard/FNO truth and t=5 anchor |
| `load_cross_resolution` | Full N64/N96/N128 20-frame fields; internally executes `load_n64_short` and removes its duplicated t=5 anchor |
| `load_fno_test_dt0p02`, each grid's M3 schedule | 46-frame history and 11 report frames t=5:0.1:6, exact coarse/fine truth and anchor |
| `load_correction_holdout` | All IDs1200–1399, seven S2 report times; selected fields checked against their new dense parent's integer frames |

原调用方的比较规则保持：M2的ID、报告时间和场使用`array_equal`；M3时间使用原`rtol=0, atol=2e-12`，场仍精确比较；S1的t=5/5.5/6场精确一致，S2两个cohort报告时间一致。补充检查覆盖S2留出ID及全部200条新dense父场。S3跨分辨率端点t=5/6已取出并记录哈希，不宣称完成其模型或验证集计算。

The recorded comparisons preserve the original callers' equality rules, with supplementary holdout-ID/dense-parent checks. All passed. Since `array_equal` alone does not distinguish signed zero, float32 bit mismatch counts are separately recorded; every such count is zero. Native FNO readers also enforce full time axes, IDs, dtypes/shapes, required metadata and finite returned fields. No tolerance was introduced or relaxed.

六个选中H5的实际SHA均与集合清单一致，运行前后未变。新集合自身相互一致，不表示它与原发布场相同；原场差异和既有U-NO、GIFT/S2、PINN失败仍保留。

All six selected H5 files matched their collection manifest before and after the calls. This establishes internal data/reader consistency, not equality with archived fields, cross-device numerical reproduction, or successful training. The earlier [full-collection audit](../full_generated_collection_20260913/README.md) and its preserved wrapper failure remain separate evidence.

只导入共享reader和真实路径解析器，未导入实验`run.py`或执行实验main。路径解析器正常导入Torch，但未初始化CUDA；未导入TensorFlow，未构建或运行模型。PINN对新生成输入的运行时限制未改变。

[result.json](result.json) is a byte-exact copy of the completed read-only check, including source/input hashes and returned-array summaries, not the arrays. [manifest.json](manifest.json) lists the two payloads. Only these three evidence files are distributed; no H5, weights, raw logs or work-only verifier code are included. Numerical experiment acceptance remains false.
