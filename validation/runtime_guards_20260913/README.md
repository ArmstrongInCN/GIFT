# Runtime guard verification / 运行环境门禁验证

**PASS within the engineering scope below; not scientific acceptance.**
The candidate with pending runtime changes based on parent commit
`77e987e0de97d93944daea4fbcce583546f575f0` was tested once on 2026-09-13.
This is not a claim that the unchanged parent commit contains these guards.
The four tested runtime files and two test files are individually SHA-256 bound
in [verification.json](verification.json).

- Default CPU regression: **173 passed, 25 skipped, 3 warnings**, plus
  **24 passed subtests**, in **27.99 s**. JUnit contains 198 testcase elements
  (173 passed + 25 skipped); its `tests="222"` attribute also counts the
  24 subtests. There were no failures or errors.
- The 311-file working-tree inventory (tracked and unignored files) was unchanged
  before/after the suite. Candidate `gift` and checkpoint-module import origins
  were checked; CUDA remained uninitialized. This was not a new full import-origin
  audit of every dependency.
- A separate actual formal U-Net `--dry-run` succeeded with the documented B7
  child environment, Torch intra-op threads 1, and no environment mutation or
  CUDA initialization. It validated the plan, metadata and pinned external source,
  not the large training file's hash. No model training or experiment output was
  produced by that dry plan.

The tests cover rejection before scientific imports for an invalid U-Net startup,
explicit nonformal `--tiny`, help without runtime imports, rejection of abbreviated
options and direct-controller bypass, and shared-journal rejection of changed
Torch thread pools or any of seven launch environment entries. Missing and empty
values remain distinct. The successful startup unit fixture stops at an import
sentinel; only the separately recorded actual dry plan proceeds to metadata checks.

Old journals are not migrated or rewritten: resume requires their own frozen
source/runtime identity, which may predate the parent commit above. The shared
identity change affects baseline and GIFT training and shared experiment-resume
consumers; PINN uses its independent journal.

本次只验收工程门禁与 CPU 回归。旧日志不补写缺失线程字段，不改写历史证据；
U-Net 的实际 dry-run 不训练，也未核验完整数据哈希。既有 U-NO、S2、PINN
数值差异及其他未完成的完整训练/数据生成验收结论保持不变，不能据此宣称
全部实验复现。原正式目录没有被本验证修改。

Raw local logs are retained outside the repository; their measured hashes appear
in the JSON. This portable folder contains no private absolute paths, raw logs,
third-party source, datasets or checkpoints. Its manifest binds only these
portable summaries, not an independently rerunnable full-training certificate.
