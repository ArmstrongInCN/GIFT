# Validation scope / 验证范围

Use the environments and fixed external sources in [SETUP.md](SETUP.md). Run
commands from the project root. Tests are separate from the full experiments:
a small test passing does not establish that a complete training budget or a
reported scientific conclusion has been reproduced.

## Source and recovery tests

In the prediction/GIFT environment, install the optional test dependencies and
run `python -B -m pytest -q -p no:cacheprovider`. Tests requiring an explicitly
configured external source or another runtime are skipped; skips are not passes.

To exercise the actual four external prediction networks on tiny synthetic CPU
inputs, set `GIFT_EXTERNAL_ROOT`, then use a **new external output directory**:

```powershell
$env:GIFT_BASELINE_TEST_ROOT = '../validation/baseline_cpu'
python -B -m pytest -q -p no:cacheprovider tests/test_baseline_control.py
```

The opt-in cases compare continuous and resumed model, optimizer, scheduler and
random state. They use two tiny epochs, not 500 formal epochs. For another test
execution, choose another output directory; existing evidence is not deleted.

In the separate PINN environment, the NumPy-only sparse-regression tests need
only `GIFT_EXTERNAL_ROOT`, not an earlier project checkout:

```shell
python -B -m unittest discover -s tests -p test_pinn_stridge.py -v
```

These cases check deterministic repetition, unchanged inputs/random state,
the independently recomputed held-out penalized objective, fixed inherited
coefficients, and noiseless unpenalized coefficient recovery. Sparse penalties
can discard true small terms; the noisy tests do not assume perfect recovery.

## Scientific results and packaged files

Use [EXPERIMENT_EXECUTION.md](EXPERIMENT_EXECUTION.md) for full numerical
experiments and [FIGURES.md](FIGURES.md) for their SVGs. Use
[CHECKPOINTS.md](CHECKPOINTS.md) to distinguish fresh training, quick checkpoint
evaluation and same-run continuation. File/hash verification checks integrity;
it is not a substitute for recomputing numerical metrics.

After Git is initialized, inspect the proposed file selection with
`python -B -m scripts.audit_repository`. Once the final files have been staged,
add `--tracked` to verify the exact staged bytes as well. These are read-only
checks: they do not stage, commit or upload anything.

代码测试、断点恢复、文件哈希、完整训练和实验指标一致性分别核验。合成数据测试
不读取已有训练权重，也不代表真实数据实验通过。测试不会自动训练全部正式模型。
