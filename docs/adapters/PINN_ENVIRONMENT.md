# PINN 独立环境、CPU 原语与 KC GPU 入口

本入口不要求安装主项目 wheel。主项目 `pyproject.toml` 的 Python>=3.10/Torch 依赖面向 GIFT 与其他模型；PINN 使用独立 Python3.8，从源码根运行 `python -m adapters.pinn_runtime`，或直接运行 `adapters/pinn_runtime.py`。`adapters/__init__.py` 延迟加载三个既有 Torch 模型导出，不会仅因导入 PINN 模块而导入 Torch。

已测试的旧 PINN 环境是 Python **3.8.20**、TensorFlow **2.10.0**、NumPy **1.21.6**、SciPy **1.7.3**、h5py **3.7.0**；本轮没有安装任何依赖。核心运行器拒绝未验证的 TF/NumPy/SciPy 版本。不是 TensorFlow1.15 二进制环境：只从官方 TF1.15 固定源码调用与原训练相同的 contrib 优化器，运行在现有 TF2.10 compat.v1 图中；不能替换为 Keras Nadam。既有 Windows 独立环境中的 KC GPU 三项有界门禁已通过；noise001/known 完整 GPU 训练随后完成，但原数值比较有2/12项失败，见下文。新机器安装与其他环境等价仍未验证。

## 外置源

`GIFT_EXTERNAL_ROOT` 指向仓库外 clone 的父目录，包含 `pinn_sr/` 和 `tensorflow115/`。请从作者处自行取得源，不将它们复制到 GitHub 或 Zenodo 包。目标目录必须原先不存在，core.autocrlf=false 保持 canonical 字节。

```powershell
git -c core.autocrlf=false clone --no-checkout https://github.com/isds-neu/EQDiscovery.git "$env:GIFT_EXTERNAL_ROOT/pinn_sr"
git -C "$env:GIFT_EXTERNAL_ROOT/pinn_sr" config --local core.autocrlf false
git -C "$env:GIFT_EXTERNAL_ROOT/pinn_sr" checkout --detach 9a20ebe6783e00bc53e3cdd1fd7ff07d2170f496

git -c core.autocrlf=false clone --filter=blob:none --no-checkout --branch v1.15.0 --depth 1 https://github.com/tensorflow/tensorflow.git "$env:GIFT_EXTERNAL_ROOT/tensorflow115"
git -C "$env:GIFT_EXTERNAL_ROOT/tensorflow115" config --local core.autocrlf false
git -C "$env:GIFT_EXTERNAL_ROOT/tensorflow115" sparse-checkout set --no-cone tensorflow/contrib/opt/python/training/nadam_optimizer.py tensorflow/contrib/opt/python/training/external_optimizer.py
git -C "$env:GIFT_EXTERNAL_ROOT/tensorflow115" checkout --detach 590d6eef7e91a6a7392c8ffffb7b58f2e0c8bc6b
```

精确源 SHA256/Git blob SHA1 在根 `external_sources.json`。TF 两个文件有 Apache-2.0 声明；PINN 所选固定上游未找到明确再分发授权，不能从引用或 clone 推断授权。本候选只含原创加载/协议 glue。

## 计划与很小的 KC NAdam 任务

在源码根运行，`PINN_PYTHON` 指向用户自己的该独立环境 Python；`GIFT_DATA_ROOT` 指向外置数据包。以下是 **8 条观测、12 条物理行、1 次 NAdam 的小测试**，不是 5000 步或完整 PINN。`--mode known` 表示 KC，`open` 的非零系数完整梯度尚未通过严格旧图比较，不能用于正式复现。

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:PYTHONHASHSEED='0'

$pinnInit = Join-Path $env:GIFT_DATA_ROOT 'auxiliary/pinn_initialization/network_initialization_tf115_seed1234.npz'
$pinnSample = Join-Path $env:GIFT_DATA_ROOT 'auxiliary/pinn_sampling/noise_001_seed1234.npz'
$pinnData = Join-Path $env:GIFT_DATA_ROOT 'm1_parameter_identification/noise_001.h5'

& $env:PINN_PYTHON -s -B -m adapters.pinn_runtime --mode known --initialization $pinnInit --sampling $pinnSample --data $pinnData --output ../gift_runs/pinn_kc_small --train-rows 8 --physics-rows 12 --steps 1
```

默认仅打印计划，不导入 TensorFlow、不建立图、不读实验数据、不创建输出。明确追加 `--execute` 才执行；首次运行输出目录必须不存在。之后用同参数、同目录再加 `--resume --execute`，其中 `--steps 1` 是**本次追加**一步，而不是覆盖累计迭代数或冒充阶段总预算。

安全门：本页的仅 NAdam 独立阶段 CLI 只允许 `known`/KC 的 `--execute`，仍仅 CPU，**没有 `--device` 参数**；`open` 明确拒绝，不能靠切换为另一种“正式训练”标记绕过。open 图/预测/梯度 API 只供另行授权的有界诊断。160/26133 元素、约2.33e-10的梯度差尚未被证明造成报告级偏差。[完整阶段控制器](PINN_PHASE_SCHEDULER.md)才是 `--device cpu|gpu` 的公开入口，默认 CPU，GPU 仅 known/KC。noise001 旧 CPU 完整预算任务已在2000步有效提交后有意暂停并保全；不是训练完成，也没有迁移到 GPU 续算。

clean 对应 `auxiliary/pinn_sampling/noise_000_seed1234.npz` + `standard_ns_n64_full_spectrum.h5`；10% 对应 `noise_010_seed1234.npz` + `m1_parameter_identification/noise_010.h5`。采样与数据的 canonical SHA 必须成对匹配；不能把 noise001 采样接到干净或 noise010 数据。默认未限制行时为原 24000 训练行和 60000 LHS+24000 训练观测的 84000 物理行。GPU全行单次更新已在下述门禁通过，但全行完整预算验收尚未完成；完整训练应使用阶段控制器，不把本页仅 NAdam 的命令当成完整流程。

固定 NAdam 超参数为 lr=0.001、physics=1、L1=1e-7、seed1234、默认 chunk32768；不提供随意更改算法以通过验收的命令行开关。所有 checkpoint 包含完整状态并绑定同一个 run，不能传入发布终点权重冒充 fresh。未训练初始化 NPZ 的生成来源及发布权利边界见 [M1.md](M1.md)。

## KC GPU 三项有界验证与独立启动

GPU02 相对保留的 GPU01 失败尝试，唯一计算修正是将 KC 外置 mask placeholder 的静态 shape 明确为 `[4,1]`。GPU01 的4项系数梯度差在这次严格对照中消除；没有改变容差、TF32/其他精度、目标、优化器或原 chunk。Open 不应用此修正，其原梯度门禁仍保留。

| GPU02 实际完成的门禁 | 范围与结果 | RESULT SHA-256 |
| --- | --- | --- |
| Gate1 | 零系数与非零稀疏mask两组 tiny KC：全部26047个梯度元素及更新参数/slots exact | `9E7E7C4485312A2FD96C0B685622C21E3C72942123C985FD260E39A5CCA779A9` |
| Gate2 | tiny完整FSM：17次NAdam、6轮STRidge、L-BFGS边界新进程续算；81变量、游标与完整科学trace exact；L-BFGS真实预算STOP，非收敛 | `C8E60EB443E130D02FF5744DB5AA9591DD0EA190D584D4A42F4CE3C4A18D514C` |
| Gate3 | 全24000观测/84000物理行、chunk32768的一次目标/梯度/NAdam更新 exact，19.396秒 | `E847105DFA10C880DFCC218F28969C105E5A5EC9F7E42EAB6623C0F2B6FA384E` |

已打包的 [GPU三门及CPU保全旁证](../../validation/pinn_gpu_20260911/README.md) 保留原GPU01失败、GPU02严格比较及旧CPU冻结源码映射，不包含权重、原始场或训练journal。

这些都是同 GPU 上的受控比较，不是 CPU→GPU 跨设备续算、完整31000次NAdam加L-BFGS训练或 M1 验收。三门所测 runtime/controller 已按原字节集成；保留的源码注释可能仍描述隔离准备时的“GPU未验证”，不能将其读成完整GPU训练已通过，实际证据范围以上表及后续CLI检查为准。

随后真实GPU公开CLI完成一条tiny新任务（rc0，17.756秒），再以同一任务执行已完成状态的 `--resume`（rc0，10.204秒）：`already_complete` 从false变为true；首条17次NAdam/6轮STRidge完成，终态提交hash正确，第二条未追加训练且整个job文件清单/字节保持exact。这不是再次17步训练，也不是中途恢复测试的替代。结果明确 `diagnostic=true`、`scientific_acceptance=false`，`TINY_VALIDATION.json` SHA为 `D845FD7DDDC8CB43B92DDA4CD9733EF3E1CB58B3E892F06E7543E96267D5BFF3`。
便携的 [候选GPU CLI tiny旁证](../../validation/pinn_gpu_20260911/candidate_cli/README.md) 单独保存这些调用与文件身份，不覆盖原GPU三门记录。

通过上述检查后，另一个全新noise001/known GPU任务已完成原完整预算和chunk32768：进程rc0、3433.60秒、31000次NAdam及6轮STRidge；源、输入、预算和终态提交校验通过。独立读取实际终态系数×mask后，原三行12个数值中**2项失败**，均为gamma派生的绝对误差与相对误差；三个参数estimate本身均通过。它从未训练初始化开始，与tiny任务及保全的旧CPU任务分别独立，没有CPU→GPU续接；完成训练不等于原数值验收通过。具体数值及L-BFGS边界见 [完整训练失败旁证](../../validation/fresh_20260910/pinn_known_noise_001/README.md)。

本次pre-NAdam5000边界的60个主状态和22个诊断累积器均exact；L-BFGS后18个网络张量及1个系数张量不同，slots/beta/mask仍相同。两边参数/梯度打包顺序相同已排除，首个分歧操作及具体原因尚未证实。另一次隔离CPU试验只把open mask补为`[90,1]`，仍保留160/26133梯度差及3个更新参数张量差；试验失败后停止，未集成，不解除open门禁。

Windows GPU 使用**独立 PINN 环境自己的** CUDA11.2.2/cuDNN8.1 DLL，不能把主 Torch cu126 环境当作同一依赖集合。以下示例由 `PINN_PYTHON` 定位用户自己的独立环境 Python，仅读取环境目录并设置新子进程 PATH；不安装 DLL，不修改系统/用户全局 PATH，不打开新窗口，不删除文件。在 clone 根运行，先设置仓库外的 `GIFT_DATA_ROOT` 和 `GIFT_EXTERNAL_ROOT`：

```powershell
$pinnPython = (Resolve-Path -LiteralPath $env:PINN_PYTHON).Path
$pinnEnv = Split-Path -Parent $pinnPython
$pinnDll = Join-Path $pinnEnv 'Library/bin'
if (-not (Test-Path -LiteralPath $pinnDll -PathType Container)) {
  throw 'This Windows PINN environment has no Library/bin; stop and check the environment.'
}
$start = New-Object System.Diagnostics.ProcessStartInfo
$start.FileName = $pinnPython
$start.UseShellExecute = $false
$start.CreateNoWindow = $true
$start.WorkingDirectory = (Get-Location).Path
$start.Arguments = '-s -B -m training.train_pinn --mode known --device gpu --condition noise_001 --output ../gift_runs/pinn_kc_gpu_001'
$start.EnvironmentVariables['PATH'] = "$pinnDll;$pinnEnv;" + $start.EnvironmentVariables['PATH']
$fixed = @{
  CUDA_VISIBLE_DEVICES='0'; OMP_NUM_THREADS='1'; MKL_NUM_THREADS='1'
  OPENBLAS_NUM_THREADS='1'; TF_FORCE_GPU_ALLOW_GROWTH='true'
  PYTHONHASHSEED='0'; PYTHONDONTWRITEBYTECODE='1'; PYTHONNOUSERSITE='1'
}
foreach ($key in $fixed.Keys) { $start.EnvironmentVariables[$key] = $fixed[$key] }
$child = [System.Diagnostics.Process]::Start($start)
$child.WaitForExit()
if ($child.ExitCode -ne 0) { throw "PINN child exited with code $($child.ExitCode)" }
```

默认仅输出计划，不导入TF或启动训练。确需新训练时，在 `$start.Arguments` 末尾追加 `--execute`；以后仅同一GPU任务、同源/同环境/同目录可再加 `--resume`。GPU与CPU任务分别使用新的独立目录，不要同时运行两者或与其它GPU实验争抢设备。`--device cpu` 是默认值并要求 `CUDA_VISIBLE_DEVICES=-1`；GPU显式要求 `0` 且只有一个可见GPU，不允许静默CPU回退。Session采用线程1、allow-growth及0.9显存比例。这里不主动改变 TF32、确定性或 oneDNN 开关；运行器读取并绑定实际profile，不能据此声称历史TF32设置已恢复或其它设备逐位等价。

旧2000步CPU任务的7份原源码、完整journal和启动记录已保全，换目录的新进程 **restore-only** 实际通过，原记录及归档均未改动；最终报告 SHA `0014F7F8C3D867F45ADB3D2548DD40E55A468CBA4C64AC91046F063FC30C041F`。恢复只读检查未执行forward、更新或save。若日后继续这条旧CPU任务，须使用它自己的冻结源码副本、原启动命令与环境；不能用当前新源码加 `--device cpu` 绕过旧identity，更不能将其接到GPU。保全源码不是整个环境的跨机器备份。

## 可调用边界与尚缺部分

`PINNRuntime` 提供 `predict`、`library`、`objective`、`step`、`state/restore`；`NAdamCheckpoints` 提供同 run 的安全保存/恢复原语。预测会执行 forward，读 JSON/NPZ 文件本身才是纯缓存读取。两模式同次两步恢复通过；KC 的非零系数、稀疏 mask、两 chunk 的完整梯度和一步更新通过。open 在同一更强检查中仍有微小但明确的梯度不一致，未放宽断言。

[STRidge](PINN_STRIDGE.md) 和 [L-BFGS 状态驱动](PINN_LBFGSB.md)已有合成/解析目标的有限验证；[KC 阶段控制器](PINN_PHASE_SCHEDULER.md)另已通过 mock、固定小预算真实全阶段联动、新进程续算及 CLI 终态导出。noise001/known完整GPU训练已完成，但原终态指标比较2/12失败，其他条件完整训练及整套M1科学验收仍未完成。M1另提供[已训练 PINN 终态的快速读出](M1.md#9-已恢复训练终态的独立-pinn-快速读取)，不运行这些训练阶段、不解除open训练门禁。本页NAdam接口不产生原公开COMPLETE、不生成完整45行表，也不解决原参考与从零结果在L-BFGS后的漂移。

维护者可以用 `GIFT_LEGACY_PROJECT_ROOT` 显式指向自己已有的只读旧图，用 `GIFT_PINN_TEST_OUTPUT` 指向新的仓库外测试目录，运行：

```powershell
& $env:PINN_PYTHON -s -B -m unittest discover -s tests -p test_pinn_runtime.py -v
```

未配置外置数据/源/oracle 或未在 TF 环境中运行会明确 skip，不可计为通过。测试输出保留，不自动批量删除；此前完整CPU对照为 **1 通过、1 失败**（open）。GPU02字节集成后的新增设备合同4项、原FSM/CLI mock10项及known CPU实图1项回归通过，最后一项21.639秒；这次没有重跑open，也不能注销其旧失败。GPU公开CLI的tiny终态与已完成任务复用现也分别实际通过，证据见上文；它们是三门harness之外的两次真实调用，不提升为完整预算验收。
