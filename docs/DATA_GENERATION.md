# 数据生成 / Data generation

本入口从明确初值重新求解 Navier–Stokes 方程；不加载模型权重，不读取已有涡量真值来充当新生成结果。它与模型的独立从零训练、模型自身续算、已有结果取数是不同操作。下载的数据包可以直接用于训练，通常无需重新生成全部真值。

This entry point freshly integrates the Navier–Stokes equation from declared initial conditions. It does not load model weights or copy saved vorticity as newly generated truth. Data simulation, fresh model training, same-run model resume, and existing-result retrieval are distinct operations. The downloaded data package can be used directly for training.

**当前生成→训练/正式评估尚未完成全量实测验收。** 下述生成与续算轻测试不等于科学验收通过。评估元数据、时间标签、组装入口及严格 released/regenerated profile 接入已实现；新数据必须经组装和显式 regenerated 门禁，不能仅移动文件或假冒原SHA。尚未全量新生成、组装或正式训练验收，详见文末“训练接入现状/限制”。

**Full real-data generation-to-training/evaluation acceptance is not yet demonstrated.** Generator metadata/time fixes, a create-only assembler and strict released/regenerated training profiles are implemented. Newly generated inputs require assembly and the explicit regenerated gate; moving files or pretending they have the released SHA is insufficient. Full-population generation, assembly and formal-budget acceptance remain untested.

## 先做小样本 / Start with a pilot

在仓库根目录执行。`/new/...` 是用户选择的、尚不存在的仓库外目录；Windows 可使用相应的盘符路径。输出不得置于代码仓库或 `GIFT_DATA_ROOT` 内。省略 `--execute` 只显示计划，不写文件或开始积分。

Run from the repository root. Replace `/new/...` with a new external directory (use drive-letter paths on Windows). Never use the code repository or `GIFT_DATA_ROOT` as an output destination. Omitting `--execute` prints the plan without writing or integrating.

```sh
python scripts/generate_data.py --dataset standard --split training --subset 0 --pilot-steps 8 --output /new/ns-pilot --execute
```

`--subset 0,2,50:52` 使用真实轨迹 ID，闭区间，不是 H5 行号。`--split` 可选 `all/training/validation/test`，组合中出现不属于所选数据集的 ID 会报错。`--pilot-steps` 从 t=0 开始，显式标记 `COMPLETE_PILOT`；不能用作完整实验输入。

`--subset` accepts global trajectory IDs and inclusive ranges, not row indices. IDs outside the selected dataset/split are rejected. `--pilot-steps` starts at t=0 and produces explicitly non-formal `COMPLETE_PILOT` output, not a complete experimental input.

## 完整协议 / Full protocols

移除 `--pilot-steps` 才使用下表全部保存时刻。每个命令只运行一个数据集；默认包含该数据集所有 split/ID，可独立运行。生成文件始终为该 attempt 目录的 `data.h5`。完成后应先验收，再由用户放入**新的**数据集合对应相对路径；程序不覆盖下载包。

Without `--pilot-steps`, the following full schedules apply. Each command generates one dataset, with its complete split/ID selection by default. Its output is `data.h5` inside the attempt directory. Validate a completed output before placing it at the corresponding relative path in a **new** data collection. The downloaded package is never overwritten.

| `--dataset` | 完整组与时刻 / Groups and saved times | 数据集合中的相对路径 / Collection-relative path |
| --- | --- | --- |
| `standard` | N64 training IDs 0–49、validation 50–69：0:0.02:10；test 1000–1199：5:0.5:10 | `standard_ns_n64_full_spectrum.h5` |
| `fno-training` | N64 training 0–49 + 1200–2149、validation 50–69：0:0.02:10 | `fno/fno1000_n64_t0_t10_dt0p02.h5` |
| `fno-training-coarse` | 同上 / same IDs：0:0.1:10 | `fair_short_horizon/fno1000_n64_t0_t10_dt0p1.h5` |
| `fno-test` | `N64/test`：4.1:0.02:8；`N96/test`、`N128/test`：4.1:0.02:6；IDs 1000–1199 | `fno/fno_test_n64_n96_n128_dt0p02.h5` |
| `cross-resolution` | `N96/test`、`N128/test`：4.1:0.1:6；IDs 1000–1199 | `cross_resolution/cross_resolution_n96_n128_t4p1_t6p0_dt0p1.h5` |
| `short-test` | N64 test：4.1:0.1:6；IDs 1000–1199 | `fair_short_horizon/standard_ns_n64_full_spectrum_test_t4p1_t6p0_dt0p1.h5` |

```sh
python scripts/generate_data.py --dataset standard --device cuda --output /new/standard-attempt --execute
python scripts/generate_data.py --dataset fno-training --initial-conditions /download/initial_conditions/baseline_extra950.json --device cuda --output /new/fno-training-attempt --execute
python scripts/generate_data.py --dataset cross-resolution --device cuda --output /new/cross-attempt --execute
```

FNO 的两种训练输出都必须提供 `--initial-conditions`。该 JSON 是 950 条四涡旋参数，不是未来解、模型检查点或训练结果。程序验证其 shape、真实 ID 和固定参数 SHA-256，然后从 t=0 积分，**不会复用前 50 条或验证集的旧真值**。

Both FNO-training variants require the extra-950 parameter JSON. These are four-vortex initial conditions, not future solutions, model checkpoints, or training results. Their shape, IDs, and parameter SHA-256 are verified before integration from t=0. No old solutions are reused for the first 50 training trajectories or the validation group.

## 本次自身续算 / Resume this attempt only

```sh
python scripts/generate_data.py --dataset standard --split training --subset 0 --pilot-steps 8 --checkpoint-every 2 --stop-after-steps 3 --output /new/resume-pilot --execute
python scripts/generate_data.py --dataset standard --split training --subset 0 --pilot-steps 8 --checkpoint-every 2 --output /new/resume-pilot --resume --execute
```

第二条必须使用相同科学参数、设备、运行时与源代码。`--stop-after-steps` 只控制本次进程何时暂停，不改变任务协议；可移除。检查点间隔也可改变。程序绑定 attempt UUID、完整参数、代码哈希、Python/NumPy/PyTorch/设备与数值环境，并只加载该目录自己的已哈希 complex64 频谱状态、积分步和批次位置。方程积分在初值生成后不再调用随机数，不需要模型/优化器检查点。已有完成目录和无 `--resume` 的已有目录会被拒绝。

Resume requires the same scientific configuration, device, runtime, and source hashes. The pause control and checkpoint frequency may change. The saved state includes the attempt identity, complex64 Fourier state, physical step, and batch position, bound to the exact parameter/configuration/source/runtime record. Integration itself makes no random draws after initial-condition construction. No model or optimizer checkpoint is involved. Completed attempts and accidental reuse of existing output directories are rejected.

`run.json` 是不可覆盖的任务说明；`checkpoint_*.npz` 保存本次频谱状态；`latest.json` 原子指向最近已提交状态。检查点之前已写的帧在续算时可能按同一状态重写；未完成 H5 的空帧为 NaN，且标记 `INCOMPLETE`。仅同时具有 `COMPLETE.json` 和完成标记的 H5 才是生成结束；中断时不要直接把不完整文件当成数据集。旧检查点保留，程序不批量删除。文件系统损坏不属于可保证恢复范围。

`run.json` is immutable; checkpoints are retained and `latest.json` is updated atomically. Resuming may rewrite frames from the same saved state. Unwritten frames are NaN and the H5 remains `INCOMPLETE`. Only a completion record together with a completed H5 marks a finished generation. Filesystem corruption is outside the recovery guarantee.

## 初值来源与数值边界 / Provenance and numerical boundaries

- 方程和离散：`src/even_full_spectrum_ns.py` 保留原字节，SHA-256 `C4064B06D940AE05BE4904012D125D7FAE3300B03F04679748D7F3F538B37A1D`；float32/complex64、ETDRK4、内部 dt=0.005、N=64/96/128 对应 padding=99/147/195，无状态频谱截断。
- Initial conditions: first five trajectories use explicit four-vortex parameters with circulation ×1.5; the additional 45 use NumPy PCG64/default_rng seed `2026080604`. A 1,150-draw evaluation pool uses seed `2026080801`: validation uses rows 0–19, test uses rows 950–1149. The exact draw order is in the new CLI.
- 该种子协议来自只读历史自有生成器的参数协议，历史源码 SHA-256 `F79B69AEBAB8E1CD55118522CF5904961C5D881074771B19E52FAA2F57F3BAA7`。未复制其目录、旧求解器或复用旧真值；本 CLI 为独立实现，运行时不访问历史工程。恢复的 training50/test200 参数与正式数据中的对应参数逐项完全相等；validation20 用 t0 字段另行核对。
- Extra 950: their parameter-array SHA-256 is `77A90CCD984690B4C5C3CA45C281AF69E59199F138FD6AA05E3902CCAC817126`. Their original seed-generation provenance is **not recovered**. This pathway is **fresh integration from specified initial conditions**, not a claim of reconstruction from the original random seed.
- 默认完整批次：standard train50/validation20/test200；FNO training 每批50；N64 test200；N96/N128 每批8。`--subset` 或 `--batch-size` 会改变实际批次形状，CPU/GPU、FFT 库、设备和版本也会影响末位数值。程序记录而不隐藏这些差异，不擅自全局覆盖 TF32 设置；相同种子不承诺跨平台逐比特一致。
- Storage differs from historical containers: new H5 includes attempt/provenance metadata and parameter arrays. Therefore whole-file hashes are expected to differ. Dataset generation completion is not equivalent to experimental numerical acceptance. Original reference tables, tolerances, and observed failures are unchanged.

## 已实测范围与缺口 / Verified scope and remaining gaps

2026-09-10 首批 CPU 轻测试：5 项通过。验证了标准参数协议、输出路径保护、N64 单轨迹 8 步积分、3 步后同 attempt 恢复与连续积分逐项 bitwise 相同，以及 N96/N128 各一条轨迹一步积分。只读比较了训练50、验证20的 t0：最大绝对差均 `1.9073486328125e-6`；extra950 首条为 `1.1814699973911047e-6`。采用**运行前确定的初值诊断范围** atol=1e-5、rtol=1e-6；不是修改原实验数值验收门限，也不是逐比特一致。尚未执行完整2000步/全样本新生成，尚未用新生成数据完成完整训练/六实验验收。

Initial CPU checks passed, but only the bounded tests above were executed. Full-duration/full-population generation and subsequent full-budget model training/evaluation using the newly generated data remain untested. The t0 diagnostic threshold is separate from, and does not alter, the original experiment acceptance criteria.

### t0 精度定位补查 / Bounded initial-state precision diagnosis

另做训练ID0、N64、CPU单线程、最多8步的有界补查：恢复的16个初值参数与原参数逐位相同；四个Gaussian解析频谱均为complex64，逐涡旋求和及zero-mode步骤与原 `build_initial_hat` 逐位相同。历史与当前求解器源SHA都为上列 `C406…`；原生成器同样直接调用该初值函数和 `torch.fft.ifft2` 保存场。对 NumPy FFT 入口安装“若调用立即失败”的测试检查后，初值及8步积分完成，**NumPy FFT调用为0**。因此这里没有采样入口所发现的 NumPy 2 单精度FFT变化，也没有证据支持给clean生成器加入float64 FFT补丁。

| 已保存边界 / Saved boundary | 最大绝对差 / Maximum absolute error | relative L2 | 非逐位相同值 / Unequal values |
| --- | ---: | ---: | ---: |
| t=0，积分前 / before integration | 1.1920928955078125e-6 | 1.66532692351771e-7 | 3843/4096 |
| step4，t=0.02 | 1.9073486328125e-6 | 2.370408662336258e-7 | 3878/4096 |
| step8，t=0.04 | 2.384185791015625e-6 | 4.1591648704048785e-7 | 3976/4096 |

最早**可观测**差异已在解析初值频谱经Torch逆FFT保存为t0时出现，早于积分、导数乘子和padding乘积。原H5未保存逐涡旋频谱、合并频谱或实际生成设备/版本，因此仅此CPU检查不能进一步断言差异首先来自 `torch.exp`/复相位、Nyquist合并、求和还是逆FFT；设备/数学库差异只是候选上游原因，不是已证明因果。本次未修改生成或求解数学、未使用原t0作积分输入，也未放宽任何验收阈值。若需继续定位，应在原生成数值环境可用时以同参数保存上述中间值逐段比较；单轨迹/单设备结果不能替代全批次与长期积分验收。

The earliest observable discrepancy is already present in the saved initial field, before time integration. Parameters match exactly, and the historical/current solver sources are identical. Clean generation uses analytic complex64 Torch spectra and Torch FFT, not NumPy FFT; the sampling-specific float64-FFT fix is therefore not applicable. Historical intermediate spectra and actual runtime details are absent, so CPU-only evidence cannot separate exponential/phase, Nyquist merge, summation and inverse-FFT backend effects. No numerical implementation or acceptance threshold was changed. The new `test_single_trajectory_initial_path_diagnostic` records these differences without asserting cross-device bitwise success.

TensorFlow 1.15 未训练初始化 NPZ **不在本生成入口范围内**，其权利仍待审定，继续排除于数据包的 CC BY 4.0 授予。下面两个独立入口只做噪声观测与采样数据准备，不复制外部网络、优化器或训练代码，不生成初始化权重。

The TensorFlow 1.15 untrained-initialization NPZ is **not generated here**, remains subject to rights review, and is excluded from the dataset's CC BY 4.0 grant. The two independent preparation commands below do not copy or run external network, optimizer, or training implementations.

## 派生噪声：保持全组统计与随机顺序 / Paired noise with original statistics

```sh
python scripts/generate_noise.py --clean /download/standard_ns_n64_full_spectrum.h5 --output /new/noise-attempt --execute
python scripts/generate_noise.py --clean /download/standard_ns_n64_full_spectrum.h5 --output /new/noise-attempt --resume --execute
```

第一条使用指定 clean 输入重新叠加噪声，同时生成 `noise_001.h5`、`noise_010.h5`；第二条只用于未完成的同一次任务。clean 可以是下载的固定输入，也可以是独立生成并完成的新 clean 数据集，不要求二者在一个进程内生成。记录明确写作“从指定清洁输入派生”，不冒称本入口重新积分了 clean。程序不读取原噪声 H5 作为生成目标；测试才会以只读 oracle 比较。

The first command derives paired noisy observations from the specified clean input; the second resumes an incomplete same-attempt generation. Clean input can be the downloaded dataset or a separately completed fresh simulation. The receipt explicitly records derivation from a supplied clean input, not reintegration of that input. Reference noisy outputs are used only by optional read-only tests, never by generation.

原协议不可改成每条轨迹各自缩放：

- **尺度为全组 `std(clean_group, ddof=0)`，不是 trajectory std。** 先按轨迹顺序逐条读完整501帧，以 float64 分别求和、平方和，再顺序累加算总体方差。
- 单一 `RandomState(0)` / MT19937；顺序为 **training → validation → 各组轨迹行 → 每轨迹16帧块**；跨轨迹、跨组都不重设随机种子。
- 同一块标准正态 `Z` 同时用于1%与10%：`(clean.astype(float64) + fraction * group_sigma * Z).astype(float32)`；不按有限样本 RMS 重新归一化，不给时间、坐标、forcing 加噪。
- 默认每32块保留一次自身检查点，可用 `--checkpoint-every` 调整；`--stop-after-chunks N` 有界暂停。完整 MT19937 624个状态字、位置、缓存高斯值，以及下一 group/trajectory/frame 块位置均保存；输入文件哈希、代码与 NumPy 版本变化时拒绝续算。

In short: use a **group-wide** population standard deviation, a single uninterrupted MT19937 stream, paired draws, float64 noise addition, and float32 storage. The full RNG state (including the cached Gaussian) and next block position are checkpointed. `--pilot` permits a reduced clean input and labels its outputs non-formal; it does not change the formal protocol silently.

CPU 实测：按上述全组统计，从指定原 clean 新生成两条件首32帧，共262,144个数值，与原噪声 H5 **bitwise exact**；未生成后续2,238块。另用小样本验证了跨 training/validation 边界的暂停恢复与连续运行 exact，及带缓存高斯值的 RNG 恢复。尚未在新入口运行两份完整噪声数据集。

The bounded oracle test matched both conditions' first 32 frames exactly. A separate reduced input verified resume across the group boundary; a cached-Gaussian test checked all RNG-state components. Full paired-noise generation was not run in this validation.

## PINN 观测采样 / PINN observation sampling

```sh
python scripts/generate_sampling.py --input /download/standard_ns_n64_full_spectrum.h5 --output /new/sampling-clean --execute
python scripts/generate_sampling.py --input /download/m1_parameter_identification/noise_001.h5 --output /new/sampling-001 --execute
python scripts/generate_sampling.py --input /download/m1_parameter_identification/noise_010.h5 --output /new/sampling-010 --execute
```

每个命令独立生成该条件的 `sampling.npz`。使用指定输入的 training ID0，固定 RandomState1234，依次取500传感器、60时间层、24,000训练行/6,000验证行、60,000 LHS点；按 sensor/time 的 C 顺序展开。先重新生成随机设计，再仅读取所选60帧重建速度并组成 `u,v,ω` 目标。可用 `--stop-after-design` 保存完整设计和 RNG 后暂停，再以相同参数加 `--resume` 完成目标计算。三个条件的设计相同，但目标来自各自的观测数据，不互相复制。

Each condition produces its own `sampling.npz`: trajectory ID0; seed1234; 500 sensors; 60 time layers; 24,000/6,000 measurement split; 60,000 stratified LHS points; sensor-major C-order flattening. `--stop-after-design` and `--resume` exercise the saved design/RNG checkpoint. Only the selected 60 input frames are read for velocity reconstruction. No model is loaded.

数值语义锁：NumPy 2.0 改变了单精度 FFT 的计算精度；本机2.2.6直接对float32做FFT会产生与原缓存不同的速度末位。新入口显式使用 float64 FFT / complex128 中间频谱，再按原协议输出 float32，以保留原缓存语义。初测差异最大 `9.536743e-7`；独立核对确认显式双精度后，三条件全部10个数组（共30数组）与原 NPZ **bitwise exact**，无需任何容差。这一变化只限采样的 NumPy Biot–Savart 计算，不改变 NS 求解器的 float32/complex64 定义。[NumPy 2.0 official release notes](https://numpy.org/doc/2.0/release/2.0.0-notes.html)

The sampling implementation explicitly preserves the original cache's double-precision FFT intermediates despite NumPy 2.x's changed single-precision FFT behavior. All 30 arrays across the three conditions then matched the reference arrays bitwise, including after design/resume. This does not change the NS solver, experiment tolerances, or claim that subsequent PINN training reproduces its published coefficients.

运行轻测试（`/new/test-temp` 必须尚不存在，避免测试工具清理旧目录）：

```sh
python -B -m pytest tests/test_data_generation.py tests/test_derived_data_generation.py tests/test_generated_data_assembly.py -v -s -p no:cacheprovider --basetemp /new/test-temp
```

Optional read-only package comparisons require `GIFT_TEST_REFERENCE_DATA_ROOT` to point to the downloaded package; otherwise that one test is explicitly skipped. Tests perform no model training and load no weights. Test outputs belong outside the code and data packages.

## 训练接入现状/限制 / Current training integration and limitations

2026-09-10 追加只读接口审计采用原函数的精确 AST 提取与约23–26 KB的稀疏H5元数据夹具；不导入模型、不积分、不训练。夹具的场数据未写入、填充值为NaN，明确标记 `METADATA_ONLY_MOCK_NOT_SCIENCE`，不能用于实验。结果说明的是接口阻断，不是科学通过：

| 消费端 / Consumer | 当前结果 / Current finding | 必须完成的最小工作 / Required minimum work |
| --- | --- | --- |
| 独立 FNO/U-NO/U-Net formal training | 已实现 `--data-profile released/regenerated`，默认仍严格核原SHA；regenerated复用组装器检查完整来源、ID/时间/PDE/参数及执行时全部哈希/有限性。 | CPU门禁测试通过；正向元数据测试的科学边界为mock，不是实际全量集合训练通过。 |
| GIFT低频/高频训练 | 已实现同名profile门禁，复用组装器；clean/noisy低模型及高分支写入自身输入绑定。高训练要求明确传入在相同standard输入上完成原正式预算的新低模型。 | 数据资格检查可读取完整标准文件作完整性校验（包括test），但不向训练损失/模型选择传入test场；完整新数据训练尚未执行。 |
| M2/M3 的 FNO native test reader | 初审发现缺少 `metadata_json`；**已修复生成端**。仅完成的非pilot完整选择才授予metadata `complete`；记录 regenerated 来源。原N64 reader 的稀疏mock接口测试通过，未使用新真实全量测试数据。 | 仍需全量真实生成后的reader与正式实验验收；不能把mock接口通过称为M2/M3通过。 |
| short-test 与 cross-resolution reader | 初审发现 `step*0.005` 与规范十进制标签在3/20帧差8.881784197001252e-16；**已修复生成端**，所有20时刻与原 `array_equal` 要求一致，原整数积分步未变。 | 仍需真实完整场验证；未改reader或实验容差。 |
| M1入口 | 每个数据文件必须在集合根 `manifest.json` 唯一声明且大小/实际SHA一致。新增组装器可生成这样的新集合；GIFT readout还会核对模型 `training_data.sha256`。 | 组装后的真实全量集合尚未实测；模型必须绑定该新数据。不能把原发布模型与不同SHA的新数据组合后宣称同源训练。 |

The original audit found a fixed-file-hash deadlock, missing FNO-test metadata, strict time-label mismatches, and missing assembly. Generator-side metadata/time fixes, guarded assembly, and explicit baseline/GIFT data profiles are now implemented. Full real-data assembly, training and downstream acceptance remain unverified. Positive controller tests mock the scientific boundary; they do not prove scientific reproduction.

### 已实现的双 profile 规则 / Implemented explicit profiles

1. `released` 默认：固定发布数据集身份，以受信发布清单核对每个输入的相对路径、大小、SHA-256；拒绝缺失、越界、重复路径或被改写文件。发布数据当前已有的严格门槛保持。
2. `regenerated` 必须显式选择：核对一个独立新集合的 `manifest.json`、`splits.json`、每个生成 attempt 的完成记录、非pilot/完整ID与时刻、dtype/shape/有限性、参数数组哈希、PDE/域/nu/forcing/dt/padding/精度与固定源码哈希。记录每个新文件自己的真实SHA，不将其认作发布文件SHA。独立标准训练50+恢复seed的验证20，以及参数JSON的额外950，都须明确来源；额外950仍不是from-seed声明。
3. profile、manifest哈希、每个输入SHA与生成来源都写入训练 attempt、续算检查点和模型产物；输入改变时拒绝续算。仅schema正确但没有生成完成证据、存在NaN未写帧或物理定义不明的文件不能放行。
4. 固定参考场抽样比较作为独立、可选的数值验证报告：预先确定ID、时刻、范数/容差与目的，并保存通过和失败；不替换新数据、不读取参考真值作为生成输入、不更改原六实验验收门。字段近似一致也不能改称H5文件哈希相同或完整复现已通过。

The default released profile retains original input SHA checks. The explicit regenerated profile validates complete collection provenance, populations/axes/physical protocol, source hashes and finite fields; training journals and products bind the actual input profile and hashes. Read-only plans do not claim full byte verification. Optional reference-field comparison is separate and never relabels regenerated files as released bytes.

GIFT高训练必须用 `--low-model /new/low/gift_main.pt`；低模型必须是相同standard文件SHA、原完整默认预算、明确未加载发布参数的新训练v3产物。历史独立训练v3无profile字段时，仅兼容已绑定原发布clean SHA且具备完整新训练/预算元数据的产物；再生输入不享受这个兼容例外。高分支自身产物记录 `training_data` 和 `data_profile`。五个实验现在均支持同一显式 `--low-model`，须与 `--gift-model seed=/new/high.pt` 配套；不指定低模型仍保持原发布模型路径行为。此为选择/绑定接口，不代表新组合已经完成六实验验收。

GIFT high training requires an explicit fresh, formal-budget low generator bound to exactly the same standard input and profile. Five experiments accept `--low-model` before creating their continuation session; pair it with the new high branch. Default published inference paths remain unchanged. High journals bind participating GIFT/runtime/training/generator sources and reject changed resume identity before derivative/RHS preparation; an interrupted unfinished preparation is recomputed, not silently reused. These engineering controls do not assert full-budget numerical success.

### 独立生成后如何组装 / Assembly after independent generation

各个生成命令可完全独立执行、独立恢复；不要求所有数据或模型一次run。核心输入可分别生成：① `standard`，② `fno-training`（seed训练50 + 显式参数950，并独立生成验证20），③ `fno-test`。短时与跨分辨率集合另用 `short-test`、`cross-resolution`、`fno-training-coarse` 独立生成；或将来增加仅从**本次新生成**的完整dense文件按整数保存步抽取coarse视图，明确记录派生关系，不从旧发布真值补帧。

The core clean inputs can be generated independently: standard, FNO training, and FNO test. Short/coarse/cross-resolution inputs use their independent commands, or a future verified integer-frame derivation from the **newly generated** dense outputs. No old reference trajectories may be used to fill missing generated frames.

已新增 `scripts/assemble_generated_data.py`，仅接受本候选独立生成 attempt 的 `COMPLETE.json` 与绑定的真实数据哈希，拒绝pilot/部分subset/缺帧/旧源码哈希；使用本页路径表在**新的**集合目录中安放其文件（复制自己的新生成文件属于打包，不是重新生成真值）。执行时扫描完整浮点场有限性，核对ID/时间/PDE/初值参数/源代码，逐文件复制后重验SHA；创建小写 `manifest.json`、`splits.json` 和各任务 `provenance/` 完成记录。已有目标目录直接拒绝，不要求改写或更新旧集合。

```sh
python scripts/assemble_generated_data.py --job standard=/new/standard-attempt --job fno-training=/new/fno-training-attempt --job fno-test=/new/fno-test-attempt --output /new/generated-input-collection
python scripts/assemble_generated_data.py --job standard=/new/standard-attempt --job fno-training=/new/fno-training-attempt --job fno-test=/new/fno-test-attempt --output /new/generated-input-collection --execute
```

第一条仅检查来源/结构并显示计划，不创建目标、不完整扫描场；第二条才执行全部输入哈希/有限性检查和字节保留复制。可增加 `--job short-test=...`、`cross-resolution=...`、`fno-training-coarse=...`、`noise=...`、`sampling-clean=...`、`sampling-001=...`、`sampling-010=...`。噪声任务的clean哈希必须等于集合内新生成standard的哈希；采样任务的输入哈希必须等于集合中对应新生成clean/noise哈希。仅有下载数据、缺少自身生成完成记录或其parent未进入本集合时拒绝，不能偷用旧真值/噪声补齐。TensorFlow初始化不纳入组装器。

`manifest.json` 明确列出 available_jobs/missing_jobs，允许只为某个独立消费者组装所需的完整文件；缺少其他任务时不声称六实验输入齐全。状态为 `ASSEMBLED_SCHEMA_VERIFIED_NOT_EXPERIMENT_ACCEPTED`，不是原发布数据，也不是数值验收通过。已做pilot/无receipt/错误父输入/未写NaN拒绝及字节保留复制轻测试；**没有全量真实组装成功测试**。基线与GIFT的 `--data-profile regenerated` 已实现，但不能用 `--tiny` 绕过正式资格门冒充正式训练。

The create-only assembler validates current generator/solver hashes, completion receipts, full selected-file populations, initial parameters, physical protocol, exact axes, and (on execution) full hashes/finite fields before byte-preserving copies. Derived noise and sampling must bind to the newly generated parent inputs present in the same collection. Missing jobs remain explicit; the collection is not labeled experiment-accepted. Guard/rejection and copy tests were run, but a full real-data assembly was not. Baseline and GIFT regenerated profile switches are implemented; `--tiny` is not a workaround for formal acceptance.
