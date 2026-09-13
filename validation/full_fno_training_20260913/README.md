# 完整 FNO 训练数据生成 / Full FNO training-data generation

**完整性 PASS；与发布参考全场数值 DIFFERENT。** 这是训练数据的完整重新积分与终态核验，不是FNO模型训练通过。没有定义或新增全场验收容差，没有更改原参考或实验阈值。

**Integrity PASS; released-reference fields DIFFERENT.** This evidence covers full training-data integration and terminal verification, not FNO model training. No full-field acceptance tolerance was defined or introduced, and no reference values or experiment thresholds were changed.

2026-09-13，CPU生成任务一次正常完成（rc=0），耗时4813.906970700016秒；随后独立核验耗时71.82799999997951秒。训练1000条（IDs0–49及1200–2149）分20批，每批50条；验证20条（IDs50–69）为第21批。所有轨迹从t=0按内部dt=0.005积分至t=10，保存0:0.02:10的501帧，N=64、float32/complex64；未使用旧解场或模型权重作为生成输入。

The completed run contains 20 training batches of 50 trajectories and one validation batch of 20. All trajectories integrate from t=0 to 10 with internal dt=0.005 and 501 stored frames at dt=0.02. The exact 21 completion messages, own attempt/binding identity, final job 21 empty-state checkpoint and source/input/output receipts are retained in [result.json](result.json). Counts of 42,000 batch advances and 2,040,000 trajectory-steps reflect the bound schedule, completion messages and final cursor—not instrumented arithmetic-operation traces. This run was uninterrupted and adds no new pause/resume experiment.

核验读取全部 **511,020帧、2,093,137,920个float32值**，包含全部1020个t0帧；新数据及参考数据均无非有限值。共有2,088,321,636个值的比特不同。float32按uint32比较（区分正负零），误差范数以float64累积；relative L2为 `sqrt(sum((new-reference)^2)/sum(reference^2))`。

All stored values were compared, not a sample. Both fields were finite. The two maxima below are tracked independently: **the relative L2 at the maximum-absolute-error frame is not the maximum relative L2**. The aggregate norm also differs from the worst per-frame norm.

| 指标 / Metric | 数值 / Value | 位置 / Location |
| --- | ---: | --- |
| 全场最大绝对误差 / Maximum absolute error | 0.9764223098754883 | training ID1816, t=10 |
| 全场最大逐帧relative L2 / Maximum frame-relative L2 | 0.024500632762808398 | training ID1384, t=10 |
| 全场整体relative L2 / Aggregate relative L2 | 0.0015389216323043382 | 全部字段 / All values |
| t0最大绝对误差 / t0 maximum absolute error | 2.105349039993598e-6 | training ID2073, t=0 |
| t0最大逐帧relative L2 / t0 maximum frame-relative L2 | 2.148236143623744e-7 | training ID1589, t=0 |
| t0整体relative L2 / t0 aggregate relative L2 | 1.7062593444241312e-7 | 全部1020个初始场 / All initial fields |

最早可见差异已在training ID0的t0、空间索引(y,x)=(0,0)出现：新值 `2.5685876607894897e-6`、参考值 `2.742137894529151e-6`。这早于时间积分，但不能单凭已存储字段定位首个有差异的内部运算。发布H5没有原生成设备、运行时或耗时属性，不推断原计算环境或因果。

The earliest observable difference is already in the initial saved field, before integration. That boundary does not identify the first differing internal arithmetic operation. The reference container does not record its original generating device/runtime/duration; no causal or cross-device conclusion is inferred.

初值来源必须区分：前50条来自显式/声明种子协议，**后950条使用已声明的参数JSON，其原始随机种子仍未知**。这950条是从指定初值重新积分，不是重现未知原种子。全部训练参数与输入及参考参数逐比特一致；验证参数与声明的种子协议一致，但参考验证组没有参数数据集，不能声称做过不存在的参考参数比较。ID、时间、shape、PDE语义及存储差异也完整保留于结果。

The extra950 are **specified published initial parameters, not reconstruction of an unknown original random seed**. Training parameter bits match both supplied inputs and reference parameters. Validation parameters match the declared seed protocol; the reference validation group has no parameter dataset. The report explicitly preserves that unavailable comparison, along with coordinate/physical-protocol checks and H5 metadata/storage differences.

生成运行时：Python3.10.19、NumPy2.2.6、h5py3.16.0、PyTorch2.10.0+cu126，实际device为CPU，记录线程/TF32等配置不变。独立核验未导入Torch/TF，不重跑积分。完整性通过不代表模型训练、完整数据集合组装、六实验或跨设备逐比特复现通过。

This directory contains only this README, the portable full result and its manifest. All 25 original verification fields—including group/t0 summaries, independent maxima and before/after hashes—are preserved. Added provenance binds the raw result SHA `0d72d666705f696129a15940021c3cf49d0db39ea1334ec33466a1527ef6451a` and the actual generation END/log receipts. UUIDs retained in the result identify this experiment and its checkpoint, not people or devices; machine paths and process IDs are excluded.

未上传H5、NPZ、权重、临时代码或66,980,199字节的逐帧CSV；该CSV的511,020行计数和SHA仍保留，逻辑文件名指向操作者归档，不表示随本目录发布。此次打包只读既有收据，没有重算。另见[数据生成协议](../../docs/DATA_GENERATION.md)。[manifest.json](manifest.json)列出两个载荷；清单自身SHA单独交付，避免循环自哈希。
