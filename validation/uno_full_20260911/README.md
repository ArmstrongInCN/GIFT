# U-NO: full training complete; reference differences retained

The current independent controller completed its original **150-epoch** budget
from random initialization, including actual continuation from its own epoch-1
journal: 1,000 trajectories per epoch, 20-step closed-loop loss, physical batch
16 (last batch 8), native upstream backward/Adam and **9,450 updates**.

The separate completed CPU terminal audit returned
`valid_full_run=true / VALID_FULL_RUN_REFERENCE_DIFFERENCES`. Source, released
data, history, optimizer, scheduler, RNG and own terminal export bindings passed.
All **36/36** exported model tensors exactly match the run's own terminal
journal. Comparison with the published terminal gives **0/36 exact tensors**;
all 150 original training-loss comparisons are nonexact. These are preserved
differences, not a new tolerance-based training gate.

Only one independent historical loss is counted per epoch. The saved candidate
per-trajectory 20-step sum is divided by 20 to compare with the original CSV's
per-trajectory/per-step column. This converts reporting units; it does not
recover the original unsaved accumulator's rounding. At epoch 150, the converted
candidate loss is `0.022853390622138977` versus `0.021599572920799254` originally.
No tolerance was added, and neither value was substituted for the other.

The resumed training process took **8,401.320 s**. Its separate initial epoch
process took **66.069 s**; the sum is approximately **8,467.389 s**, excluding
pauses and audits. The resumed-process duration is not the whole fresh run's
duration. Actual same-run continuation is established; an uninterrupted versus
resumed full-run bitwise comparison was not performed. The observed environment
is recorded without inventing missing historical runtime metadata.

This record establishes a complete, source-bound fresh training result, **not
M2 numerical acceptance or whole-project reproducibility**. M2 evaluation of the
new terminal is separate. Existing published weights/catalogue are unchanged.
The [historical first-epoch evidence](../uno_epoch1_20260911/README.md) is retained
unchanged and remains limited to that earlier boundary.

完整 150 轮与自身 GPU 断点续算已经实际完成，终态完整性核验通过；自有导出与
终态检查点 36/36 张量逐位相同，但与原发布权重为 0/36，150 个原训练损失均不逐位
相同。保留差异与原生反向警告，不修改算法、数据或门槛。新终态能否通过 M2 数值
检验须由独立实验判断，不能把“完成训练”写成“报告复现验收通过”。

[verification.json](verification.json) retains all 150 scalar comparisons, the
36 tensor-difference descriptors, source/runtime/input identity and hashes of
the original CPU audit and invocation evidence. [manifest.json](manifest.json)
binds these portable files. Packaging read existing JSON and small source files
only: no tensor checkpoint, dataset, model import, training or inference. No
weights, private absolute paths/UUIDs or third-party source are distributed here.
