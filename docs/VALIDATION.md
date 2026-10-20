# Validation scope / 验证范围

当前可移植复现流程见 [CURRENT_REPRODUCTION.md](CURRENT_REPRODUCTION.md) 和 [W3_REPRODUCTION.md](W3_REPRODUCTION.md)。原数据/模型/数值文件的完整性要求逐字节相同；数值复算按dtype、绝对尺度、相对范数及失败mask评估计算环境差异。有限的CPU/GPU浮点差异不自动构成科学异常，也不能用宽松阈值掩盖错时间轴、错模型或缺失恢复状态。原实验记录始终保持不变。

仓库外部算法边界及目录整理的预测、权重、结果包一致性证据见
[EXTERNAL_REPOSITORY_POLICY.md](EXTERNAL_REPOSITORY_POLICY.md)。手工下载与适配步骤见
[EXTERNAL_MANUAL_ADAPTATION.md](EXTERNAL_MANUAL_ADAPTATION.md)。

## W1/W2 验证范围

[N300/76点协议](W1_W2_PROTOCOL.md) 的验证对象包括固定有序N300/N50 ID、20条validation、真实76节点精抽、单真实初态、每方法声明的seed与checkpoint身份，以及终态或validation-best规则。U-NO/U-Net的20/4步目标、batch/time-sum loss、尾批、optimizer/scheduler与随机状态对应其固定配方；W1/W2的GIFT-Lite使用自己的N300协议生成元/分支文件（来源键`GIFT-N50`），不与辅助实验的N1000协议同名权重混用。

确认与探索人口分开，每项保留完整180条轨迹及失败原因。两个人口的成绩已被查看，不视为全部方法盲测。W2核对三个网格逐元素相同的float64初值参数和真实0.04秒时间节点；仅N64训练，无目标网格统计拟合，ANIE不在W2。FNO-3D保持75时间查询轴，评价其前25点；N64的W2数组对应W1一秒前缀，失败判定也限于该前缀。

模型、数据和源文件SHA绑定见 [CHECKPOINTS.md](CHECKPOINTS.md) 与相应结果清单。六个既有模型及W2的完整预测重算入口是 `scripts.audit_w1_w2_single`，检查14个W1方法/人口测量和36个W2方法/网格/人口测量；文件检查入口为 `scripts.verify_published_results`。数值范数重算、模型身份、文件字节完整性和实际图像显示是不同验证层次，不能互相代替。

CPU测试包括 `test_w1_extended_baselines.py`、`test_w2_single_evaluate.py` 与single/profile/queue测试。可用 `CUDA_VISIBLE_DEVICES=-1` 隔离GPU，并配置公开源码的external roots。小型shape、一步或恢复测试检查实现接口，不代表完整500轮训练或正式预测性能；合成fixture不进入正式人口指标和耗时。

ANIE的时空token逐项对应与训练配置由 [ANIE实现说明](ANIE_ADAPTATION.md) 声明，其完整训练/数值审计来自 `prediction_aligned` 的 `FINAL_TRAINING_AUDIT.json`、`FINAL_RESULT_AUDIT.json`。

结果证据见 [W1来源清单](../results/formal/W1_flow_prediction_20261007/published.json)、[W2来源清单](../results/formal/W2_cross_resolution_20261007/published.json) 及声明外部目录中的 `FINAL_RESULT_AUDIT.json`、`FINAL_TRAINING_AUDIT.json`。独立W3/S1–S3和N1000辅助预测使用各自模型与数据，以下测试按其声明范围解释。

Use the environments and fixed external sources in [SETUP.md](SETUP.md).
External tests also require the user-prepared interfaces outside GIFT selected
by `GIFT_EXTERNAL_ADAPTER_ROOT`; they are not supplied by the repository. Run
commands from the project root. Tests are separate from the full experiments:
a small test passing does not establish that a complete training budget or a
reported scientific conclusion has been reproduced.

The full-data GIFT tests check trajectory-epoch sampling, train/validation/test
identity separation and exact small CPU generator continuation across phase
boundaries. Branch-controller tests cover optimizer, scheduler and RNG recovery
with a bounded surrogate; they are not full native-rollout performance evidence.
Canonical data tests check lossless row copying, metadata ranges, per-trajectory
storage chunks, read-only input checks and correct binding of GIFT versus GIFT-Lite.

## Source and recovery tests

In the prediction/GIFT environment, install the optional test dependencies and
run `python -B -m pytest -q -p no:cacheprovider`. Tests requiring an explicitly
configured external source or another runtime are skipped; skips are not passes.

To exercise the actual four external prediction networks on tiny synthetic CPU
inputs, set `GIFT_EXTERNAL_ROOT` and `GIFT_EXTERNAL_ADAPTER_ROOT`, then use a **new external output directory**:

```powershell
$env:GIFT_BASELINE_TEST_ROOT = '../validation/baseline_cpu'
python -B -m pytest -q -p no:cacheprovider tests/test_baseline_control.py
```

The opt-in cases compare continuous and resumed model, optimizer, scheduler and
random state. They use two tiny epochs, not 500 formal epochs. For another test
execution, choose another output directory; existing evidence is not deleted.

In the separate PINN environment, the NumPy-only sparse-regression tests need
`GIFT_EXTERNAL_ROOT` and the prepared `GIFT_EXTERNAL_ADAPTER_ROOT`, not an earlier project checkout:

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
add `--tracked` to limit the audit to tracked files. These are read-only
checks: they do not stage, commit or upload anything.

代码测试、断点恢复、文件哈希、完整训练和实验指标一致性分别核验。合成数据测试
不读取已有训练权重，也不代表真实数据实验通过。测试不会自动训练全部正式模型。
