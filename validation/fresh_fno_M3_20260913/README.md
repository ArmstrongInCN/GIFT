# M3: fresh-FNO inference evidence

Fresh GIFT and both fresh FNOs; M3 contains neither U-NO nor U-Net.

Inference exited successfully. Original CSV numeric comparison: **PASS**. All 1870 rows, 9350 scalar comparisons and 3740 count comparisons are retained. The complete CSV is copied byte-for-byte; every failure remains in comparison.json. Tolerance is unchanged: abs(delta) <= 0.0005 + 0.05 * abs(reference); counts must match exactly.

This packager reads SHA-bound JSON/CSV metadata and the two owned pinned stdlib helpers only. It does not read/revalidate weights, the input-bundle PT files, datasets, raw predictions or logs, and does not train or infer. Input hashes are recorded launch/audit evidence, not new physical-file verification. Prior failed results and historical weight differences remain evidence. Neither successful execution nor this CSV comparison establishes whole-project/all-model from-zero acceptance.

本包逐字节保留本次 CSV 和全部比较失败；完整训练审计、数值比较和全项目验收分开。不修改参照值或容差，不将旧 U-NO 的失败隐藏，也不重新验证模型、数据或预测数组。
