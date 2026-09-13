# Fresh-environment CPU validation / 新隔离环境验证

Scoped result: **PASS** for a new Python 3.10.19 venv and the actual GitHub
clone at commit `c6ae4d0268f782d64603bab29049b70e87a95b2f`.

The venv does not share system site-packages, but it uses an existing Python
installation. This is not a new-OS, GPU, TensorFlow-environment, or full-training
acceptance test.

## What actually ran

- Bootstrap installation succeeded; all 32 reviewed runtime versions and
  pytest 9.1.1 matched, with dependency origins inside the new venv.
- The complete Torch 2.10.0+cu126 wheel was size/SHA-256 verified before
  installation. The runtime installation succeeded in 668.7178874 seconds;
  pinned direct wheel URLs do not constitute an offline-install guarantee.
- The first editable installation failed at a 262-character Windows temporary
  destination path. Its logs and original installer were retained. A second
  attempt succeeded in 1.8805416 seconds using a short, new TEMP/TMP directory,
  with no tracked project-source or system long-path-setting change.
- Both `pip check` calls passed. The default CPU suite ran **once**:
  **149 passed, 25 skipped, 3 warnings in 21.77 seconds**. Its 24 separately
  reported passing subtests are not added again to the 149 tests.
- JUnit contains 174 testcase records, no failures or errors. The CPU child took
  50.4318209 seconds including imports and surrounding checks, not just pytest.
- All 306 tracked clone files matched their committed Git blobs before and
  after validation; HEAD and validation pins were unchanged. Seventy-five clone
  code files executed, with zero origin violations. CUDA was not initialized
  and TensorFlow was not imported.

This evidence describes the tested commit, before later documentation/evidence
additions. Editable-install egg-info was generated only in the temporary clone;
the formal original and release candidate were not installation targets.

## 范围与保留事项

已完成新建隔离 venv 的实际安装和一次默认 CPU 测试；32 项运行依赖及
pytest 的版本、导入位置、前后依赖检查和源码完整性均通过。该 venv 基于
现有 Python 3.10.19，不与原环境共享 site-packages，但不是全新操作系统。

首次 editable 安装的 Windows 长临时路径失败、此前下载/安装失败记录均
保留，不计为成功。使用较短的独立临时目录后安装成功，没有修改项目源码
或系统长路径设置。25 项跳过测试未被算作执行通过，24 项子测试不重复计数。

这里没有重新训练模型或重新验收 GPU、PINN 独立环境及检查点。U-NO、
S2 和 PINN 的已知数值差异仍未解决，本次通过不代表实验整体 100% 复现。

## Files

`verification.json` contains the portable results, all 32 runtime versions,
retained failure details, and SHA-256 bindings to original local receipts.
`manifest.json` binds the two portable files; it does not hash itself.
Personal absolute paths, wheel/package source, data and checkpoint files are
not included. Packaging performs no additional tests. Raw local evidence is
not embedded, so these hashes alone are not independent execution proof.
