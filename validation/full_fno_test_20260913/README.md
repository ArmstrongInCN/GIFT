# Full FNO-test CPU generation / 完整 FNO-test CPU 生成

Integrity: PASS. Field comparison: DIFFERENT. Scientific acceptance: NOT ASSESSED.

本次独立 CPU 任务完整生成 N64、N96、N128 的 FNO-test 数据：每个网格均为
ID 1000–1199 的 200 条测试轨迹，共 600 个轨迹—分辨率组合、77,600 个保存帧、
652,083,200 个 float32 值。生成从声明的初值参数积分，不使用旧解场或模型权重。
独立只读校验随后比较全部保存值；两侧非有限值均为 0，但 651,901,954 个值的
float32 比特不同。完整性通过不等于场一致，也不等于数值验收通过。

The independent CPU run completed the full held-out FNO-test population at all
three grids. The verifier subsequently compared every stored scalar with the
read-only released reference, including signed-zero bits. Integrity passed;
fields differ. No full-field acceptance tolerance was defined or introduced.

## Coverage and execution / 覆盖及执行

| Grid | Trajectories | Native batch | Integration steps | Stored frames per trajectory | Stored times |
| --- | ---: | ---: | ---: | ---: | --- |
| N64 | 200 | 200 | 1,600 | 196 | 4.1–8.0, every 0.02 |
| N96 | 200 | 8 | 1,200 | 96 | 4.1–6.0, every 0.02 |
| N128 | 200 | 8 | 1,200 | 96 | 4.1–6.0, every 0.02 |

The internal step is 0.005. The bound schedule, 51 batch-completion messages and
final own cursor account for 61,600 batch advances and 800,000 trajectory-resolution
steps; these are not an instrumented trace of every internal arithmetic operation.
This was one uninterrupted invocation, with no artificial pause or resume.
Its final checkpoint has job index 51 and an empty complex64 state marker.
Intermediate checkpoints were not opened or replayed by the verifier.

Generation ran from 2026-09-13 13:12:39.928484 UTC to 14:27:50.314699 UTC, returned 0,
and took 4,510.378191599972 seconds. The independent audit ran from
14:30:36.371990 UTC to 14:31:04.265954 UTC and took 27.889999999955762 seconds.
The recorded generation profile was Python 3.10.19, NumPy 2.2.6, h5py 3.16.0,
PyTorch 2.10.0+cu126, device CPU, intra-op threads 1 and interop threads 16.
The verifier imported neither Torch nor TensorFlow and did not reintegrate or train.

三个网格的全部 float64 初值参数同时与生成文件、参照文件及独立 PCG64 种子重建
逐比特相符；参数来源为 seed 2026080801 的 1,150 组参数池中 rows 950:1150。
完整 ID、时间轴、形状、dtype、有限性、物理协议、求解器元数据、来源及自身任务绑定
均通过记录中的检查。检查前后相关源码、输入、输出、终态检查点和收据哈希均未变。

## Full-field differences / 全保存场差异

Relative L2 is sqrt(sum((generated-reference)^2) / sum(reference^2)), with float64
accumulation. Aggregate relative L2 uses every stored value in that group; the
all-grid aggregate weights scalar values, not grids equally. Maximum absolute
error and maximum per-frame relative L2 were tracked independently.

| Grid | Bit mismatches / values | Maximum absolute error | Independent maximum frame relative L2 | Aggregate relative L2 |
| --- | ---: | ---: | ---: | ---: |
| N64 | 160,532,177 / 160,563,200 | 0.18331849575042725 | 0.00773763582648185 | 0.0007138084677094552 |
| N96 | 176,897,476 / 176,947,200 | 0.07293868064880371 | 0.0015633041796427243 | 0.0002888742719511077 |
| N128 | 314,472,301 / 314,572,800 | 0.07412528991699219 | 0.0015616481222031668 | 0.0002945565429390893 |
| All stored values | 651,901,954 / 652,083,200 | 0.18331849575042725 | 0.00773763582648185 | 0.0004222264111280865 |

For this full stored interval, each grid's independently measured absolute and
relative maxima happen to occur in the same frame: ID 1118, t=8.0 at N64;
ID 1118, t=6.0 at N96 and N128. This coincidence does not make the relative error
of an arbitrary maximum-absolute-error frame a maximum relative error.

These are descriptive measurements, not model-prediction errors or acceptance
thresholds. No original experimental tolerance, precision or reference value
was changed.

## First saved frames are not t0 / 首帧不等于 t0

Integration begins at t=0, but every stored comparison begins at t=4.1.
There is no stored t=0 field in these test files, and no t=0 field was compared.
The earliest observable stored-field difference, ordered by time, trajectory ID
and numeric grid, is N64 ID 1000 at t=4.1, frame 0, spatial index (y=0, x=0):
generated 2.75447416305542 versus reference 2.7544593811035156.
This frame has maximum absolute error 0.0034633874893188477 and relative L2
0.0001063524566691349; it does not locate an earlier differing solver operation.

The first-frame-only summaries also retain independently tracked maxima:

| Grid, t=4.1 only | Maximum absolute error (ID) | Independent maximum frame relative L2 (ID) |
| --- | --- | --- |
| N64 | 0.009837865829467773 (1089) | 0.0002264290608323967 (1155) |
| N96 | 0.010113000869750977 (1089) | 0.0002232392884400819 (1089) |
| N128 | 0.010182380676269531 (1089) | 0.00022852272635953198 (1155) |

最早保存边界为 t=4.1；不能将本次结果称作 t0 比较，也不能据此定位首次不同的
内部浮点运算。首帧表分别保留最大绝对差和独立追踪的最大单帧相对 L2，避免混用。

## Portable evidence and limits / 便携证据及边界

[result.json](result.json) preserves the complete original audit summary
byte-for-byte, including full-field and first-frame statistics for every grid,
independent absolute/relative extrema, metadata inventories, storage contracts
and before/after file hashes. Its SHA-256 is
8a2b540261f20c80bff4ec133e867c879b50c95efe175e34ad062e8875061bc8.
The fixed verifier SHA-256 is
8655e66c067c314c0c86ff96829e2f9b402849a04584be2408808bfc7a4d4f37.
[manifest.json](manifest.json) binds the result and this README.

The original result contains no machine filesystem paths or machine UUIDs.
Its retained attempt UUID and checkpoint filename identifier identify this
experiment and its own final artifact; they are not device or account identifiers.
Relative attempt/reference/source labels identify hashed evidence outside this
bundle and are not bundled files or download locations. H5 paths such as "/"
identify objects inside the compared containers, not a host filesystem root.

The local 77,600-row CSV is not included: 10,340,987 bytes, SHA-256
36a156730b58319658ca1e69b961a5c4feb85be0b4edcd92a0fb880485147edb.
Only README.md, result.json and manifest.json are distributed here. No H5 data,
parameter arrays, spectral checkpoints, per-frame CSV, execution logs, verifier
code, or external source code is included. Packaging performed no reintegration,
new field comparison, training or checkpoint replay.

完整摘要中的实验任务 UUID 和自身检查点标识保留用于证据关联，非机器标识。
本次便携化未重新计算；本目录仅有三个文件。完整性 PASS、字段 DIFFERENT 与
科学验收状态分别保留；未定义全场验收阈值，未证明跨设备逐比特一致、完整再生
集合、再生数据模型训练或六实验通过。生成 H5 的元数据和分块也与参照容器不同，
因此不声称整文件哈希一致。其他既有数值失败不因本记录改变。
