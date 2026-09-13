# 300-file offline index-export check

The actual 300-file staged snapshot (215,907,464 bytes) was exported and checked
byte-for-byte against both Git blobs and candidate working files. In the existing
Python environment, its default CPU suite passed: **149 passed, 25 skipped,
3 warnings, 21.78 s**. The separately reported 24 subtests are not added again.
JUnit contains 174 records, no failure/error and 25 skips. Seventy-five exported
source files executed with zero origin violations, no CUDA initialization and
no TensorFlow import. Source, export and index bytes were unchanged during testing.

Checkpoint distribution verification **failed**: 15 of 16 files passed; the
U-NO Release asset was missing. It was not inserted into this export. This is
an existing-environment offline snapshot check, not a clean installation,
GitHub clone, complete checkpoint distribution or scientific acceptance proof.

This snapshot predates this evidence directory and the accompanying status-note
update. The original execution result, file inventory and harness are SHA-bound
in `verification.json`; this portable summary does not rerun or expand that test.

这是固定 300 文件快照的离线导出与既有环境 CPU 测试，不是干净环境或 GitHub 克隆验证。
U-NO 分发文件缺失、25 项跳过及已知科学实验差异均保留；不能据此宣称全项目通过。
