# Actual private GitHub clone / 私有 GitHub 实际克隆回验

Commit `0a9848bea07c8fd7f21f2a44069f28e41c5d091e` was obtained by an actual
GitHub HTTPS clone and fast-forward fetch, not copied from the local candidate.
All 303 committed files (215,915,082 bytes) matched their Git blobs exactly.
The remote tree also matched the local commit's paths, modes, sizes and object IDs.

The separate U-NO reference was downloaded with authenticated GitHub CLI.
Its measured size and SHA-256 matched the catalog and GitHub asset digest.
After placing that downloaded file in the clone, **all 16 checkpoints passed**
byte-integrity verification. A plain clone still requires this extra download.

The existing-environment default CPU suite had **149 passed, 25 skipped and
3 warnings in 21.48 s**, with 24 subtests reported separately, not added again.
JUnit contains 174 testcase records, zero failures and zero errors. Seventy-five
clone code files executed, with no source-origin violations, CUDA initialization
or TensorFlow import. Inputs, HEAD, tree, index and the validator were unchanged.

The initial verification attempt stopped before collecting tests because its
temporary Windows write guard rejected the null device used by pytest logging.
That failed attempt is retained; only the temporary guard was corrected before
the successful second attempt. No project source, tests or acceptance threshold
were changed to obtain this result.

This evidence concerns the specified commit, which predates this summary and
its status-document update. It is **not** fresh-environment installation, full
training, or scientific acceptance. The known U-NO, S2 and PINN differences
remain unresolved. See [machine-readable scope and evidence hashes](verification.json).

本次从私有 GitHub 实际获取项目并单独下载 U-NO 附件，303 个 Git 文件与提交内容
逐字节一致，16 个检查点全部通过完整性校验。现有环境下 CPU 测试 149 项通过、
25 项跳过；不重复计算另列的子测试。首次临时验证脚本的空设备兼容错误已单独保留，
未修改项目代码或测试。本记录不证明全新环境安装、全量训练或全部实验数值通过。
