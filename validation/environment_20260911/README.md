# CPU test evidence / CPU 测试证据

The final combined default suite ran against the candidate source checkout in
the **existing** Python 3.10.19 environment: **139 passed, 25 skipped, 3 warnings,
24 subtests passed in 32.07 s**, exit 0. Subtests are reported separately, not
added to 139; JUnit's 188 records also include them. The three warnings concern
CPU-only `pin_memory` use.

The eight new staged-byte tests are included in those 139 passes. Real new
temporary Git indexes verify CRLF normalization rejection, `* -text` preservation
with SHA1/SHA256, dirty or missing work files, and inventory mismatches. They
check that the production checker leaves index and work bytes unchanged; they
do not operate on the candidate Git index. Opt-in data/runtime tests were unset,
CUDA was hidden, CPU thread counts were two, pytest cache was off, and logs,
JUnit and fresh temporary directories remained outside the source checkout.

默认 CPU 全套测试实测 139 项通过、25 项跳过，另有 24 个子测试通过；新增的 8 项
暂存区字节测试包含在 139 项内，不重复计数。使用已有环境，不能宣称新环境安装、
GitHub clone、完整模型训练或正式实验验收通过。

An **earlier, separate** portable test exported exactly 187 proposed Git files
(213,097,962 bytes) and passed 131 tests with 25 skips. Its copied imports and
bytes were checked, but it predates the new staged-byte tests and later evidence.
It is a working-file selection copy, not a staged-blob checkout or a clean clone.
Its original checkpoint verifier deliberately remains **FAIL / exit 1**:
15 of 16 artifacts passed and only the undistributed U-NO terminal weight was
missing. That outcome is not rewritten as a complete-distribution pass.

此前 187 文件导出测试与本次源码测试分开记录；不能将旧导出快照升级为当前完整
候选的便携测试。U-NO 尚缺的分发文件、干净安装与远端 clone 缺口仍保留。

[verification.json](verification.json) records exact counts, tested checker/test
hashes and original local log/JUnit/inventory hashes without publishing machine
paths. The local original outputs were retained rather than rewritten.
