# Default CPU suite after GPU02 source integration / 集成后默认 CPU 回归

One invocation from the actual candidate source, using the existing Python
environment: **143 passed, 25 skipped, 3 warnings, 24 separately reported subtests
passed in 17.72 s**, exit 0. Process wall time was 18.837 s. This JUnit actually
contains 168 testcase records (zero failures/errors, 25 skips); do not add the
24 subtests again to the 143 passed count or infer another XML record count.
The warnings concern CPU `pin_memory`, not failed tests.

本次只运行一次默认 CPU suite，新外部 basetemp 原先不存在。所有 GIFT/PINN
可选实测开关均未启用，25 个跳过项如实保留；没有执行真实 PINN TF/GPU 检查或
完整实验训练/推理任务。默认 suite 的 CPU 小夹具计算不构成科学验收。七个绑定
源码/测试文件在执行前后 SHA 一致。完整结果、依赖版本、
执行参数和原始日志 SHA 见 [verification.json](verification.json)。

This is an additional record. The [earlier run](../README.md) remains unchanged:
its 139-pass suite and historical 187-file portable snapshot retain their own
scope, including the snapshot's 15/16 artifact check and missing U-NO asset.
Neither that earlier file selection nor this direct-source run is a current
full GitHub clone, a clean installation or scientific full-budget acceptance.

Machine-local paths are replaced by symbolic roots. Original execution/log/XML
hashes are retained; large logs, fixtures, source copies and model/data files
are not repackaged here. No candidate Git staging was performed by this run.
