# External U-Net adapter

Use [Rui1521/Turbulent-Flow-Nets](https://github.com/Rui1521/Turbulent-Flow-Nets/tree/229da3e01acefa6f4dd6c77244da5015c9b965af)
at commit `229da3e01acefa6f4dd6c77244da5015c9b965af`, file `Baselines/U-net.py`.
No license was identified at the pinned source; this is not a grant to
redistribute it. The reader obtains the external source from its owner and
determines applicable permissions. No model definitions or upstream source are
included here, including the unused duplicate U-Net class in the old project.
在仓库外取得固定提交，随后在本仓库根执行检查命令；`--forward` 仅用合成输入做一次 CPU 前向。上游源码不随包发布，来源和许可边界不能由模型加载成功代替。

```powershell
$env:GIFT_EXTERNAL_ROOT = 'D:\gift_external'
git -c core.autocrlf=false clone https://github.com/Rui1521/Turbulent-Flow-Nets "$env:GIFT_EXTERNAL_ROOT\unet"
git -C "$env:GIFT_EXTERNAL_ROOT\unet" config core.autocrlf false
git -C "$env:GIFT_EXTERNAL_ROOT\unet" checkout --detach 229da3e01acefa6f4dd6c77244da5015c9b965af
python -B -m adapters.check --model unet --forward
```

Run from the release root; use the corresponding `export` syntax on POSIX.
The file's exact LF-byte hash is checked against `external_sources.json` before
import. The adapter imports `U_net` directly from this external file and calls
`U_net(input_channels=46,output_channels=1,kernel_size=3,dropout_rate=0)`.
There are 24,971,551 parameters, input `[B,46,64,64]`, output `[B,1,64,64]`.
No architecture, numerical operation or initialization is patched.
The old archive's CRLF-byte hash is retained as `legacy_crlf_sha256` for
traceability, not accepted as a second runtime variant. The canonical LF hash
was calculated from the same source with line endings normalized only.
The official checkout was subsequently checked against that LF hash.
On 2026-09-10, strict loading of the published 60-state terminal artifact,
parameter count and one finite synthetic CPU forward passed (2.58 seconds).

## Training boundary and precision

The independent training CLI is `python -B -m training.train_unet`. The formal
training is 500 terminal epochs, seed 0, batch 20, four-step closed-loop
objective, no teacher forcing/detach, no gradient accumulation, Adam lr 0.001
and weight decay 0.0001, StepLR(100,0.5), and 25,000 updates. The schedule uses
91 possible chosen starts (0,5,...,450), one per trajectory per epoch.

The earlier independent campaign, not this candidate's new controller, matched all 500 scientific
log rows and all 60 terminal state tensors exactly. It must not be retrained just
to test this loader. The verified training profile had cuDNN TF32 enabled,
matmul TF32 disabled, cuDNN deterministic enabled/benchmark disabled, strict
deterministic algorithms, and `NVIDIA_TF32_OVERRIDE` **completely unset**.
Forcing all TF32 off is not equivalent to that published training profile.
The override must not be set to `0`, `1`, or an
empty string. The adapter itself sets none of these flags. The controller sets
the backend booleans, but currently imports Torch before its profile function,
rejects only override value `0`, and does **not** implement the complete B7
pre-import environment gate. Recording a runtime is not enforcing that gate;
the documentation below does not claim a code fix.

The earlier controlled U-Net first update used physical batch 20 and **two** recursive
steps (diagnostic truncation; the formal objective has **four**). It matched
initialization, sample/window ordering, loss, 36 gradient tensors and 60 updated
state tensors against the original functions. It bound controller SHA-256
`78C3C3364785A3DA2C5710EBCC3DA065175039093AF5786527A42EAC6423525A`,
not the current controller
`7CDA8F5EE54EA16675464CD08B53370B7D309B550C6920B92F09C6DE6168999C`.
The unchanged adapter SHA is
`B79B8D750D80807F98CD118A0F3ABBF3C3DB3FF4FA3FBCC33B2BFE0ADE163524`.
That earlier record remains a two-step diagnostic and is not relabeled.

On 2026-09-11, the [actual formal four-step first-batch bridge](../../validation/unet_formal4_20260911/README.md)
separately bound current controller `7CDA...` to the actual B7 original control:
two fresh seed-0 processes, physical batch20, four recursive steps, one native
Adam update, no trained initialization. All14 fields matched exactly, including
all4 forward outputs,36 gradient tensors,60 updated states, and complete
Adam/StepLR state; the B7 thread-1 and pre-import precision profile was retained.
Original/current child elapsed times were5.110/5.085 seconds, both exit0.
This is not a full CLI/epoch/500-epoch or evaluation acceptance, and does not
add the missing pre-import gate to the candidate CLI itself.
历史B7的500轮验收与当前 `7CDA...` 的四步首批次桥接均保留、各自独立；
不能据此改标当前控制器已完成500轮。见[当前证据范围](../REPRODUCIBILITY_STATUS.md)。

### U-Net-only child-process environment

The following settings reproduce the fixed/absent environment entries actually
checked by the successful B7 preflight, including its shared TensorFlow entries;
they do not imply that U-Net uses TensorFlow. That recorded profile used
Python 3.10.19, Torch 2.10.0+cu126, CUDA 12.6 and an RTX A5500 Laptop GPU.
Other hardware/software is not thereby certified bitwise equivalent.
The evidence identifier is B7 manifest
`B7C5936A70ACD0C74C655767083876D590B3C248D9EA84BF30F2A0B890559A05`,
component manifest
`993C7FF351CB902B6472D3E1DD81A6E9F984F2BC7AB2001747E31947D3B27973`,
field `training_preflight.stdout.environment_contract`.

Run from the clone root after selecting the intended Python environment and
setting `GIFT_DATA_ROOT` / `GIFT_EXTERNAL_ROOT`. These examples default to
`--dry-run`; replace only the final arguments with
`--run-training --device cuda --output ../gift_runs/unet_001` for an explicitly
chosen new run, or that same command plus `--resume` for its own journal.
Keep the environment identical for initial training and continuation.
Do not turn off cuDNN TF32. **This is U-Net's thread-1 profile, not the tested
FNO-2D thread-16 profile.**

下面只清理/设置新子进程的环境，不修改系统或用户全局环境，不删除文件；父终端
设置保持不变。默认仅检查计划。环境准备不等于当前 CLI 已实现或通过完整 B7
preflight，也不等于已完成训练；cuDNN TF32 必须保持开启。

PowerShell (creates a child process without opening a new window):

```powershell
$start = New-Object System.Diagnostics.ProcessStartInfo
$start.FileName = (Get-Command python -CommandType Application).Source
$start.UseShellExecute = $false
$start.CreateNoWindow = $true
$start.WorkingDirectory = (Get-Location).Path
$start.Arguments = '-s -B -m training.train_unet --dry-run'
$absent = @(
  'PYTHONHOME', 'PYTHONPATH', 'PYTHONPYCACHEPREFIX', 'PYTHONSTARTUP',
  'PYTHONINSPECT', 'PYTHONUSERBASE', 'PYTHONWARNINGS', 'PYTHONOPTIMIZE',
  'PYTHONSAFEPATH', 'PYTHONCASEOK', 'CUDA_VISIBLE_DEVICES',
  'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', 'TF_XLA_FLAGS', 'XLA_FLAGS',
  'KMP_AFFINITY', 'KMP_BLOCKTIME', 'KMP_DUPLICATE_LIB_OK', 'KMP_SETTINGS',
  'ATEN_CPU_CAPABILITY', 'NPY_DISABLE_CPU_FEATURES', 'MKL_CBWR',
  'MKL_ENABLE_INSTRUCTIONS', 'CUDA_DEVICE_MAX_CONNECTIONS',
  'TORCH_CUDNN_V8_API_DISABLED', 'CUBLASLT_WORKSPACE_SIZE',
  'TF_USE_LEGACY_KERAS', 'NVIDIA_TF32_OVERRIDE'
)
$fixed = @{
  PYTHONHASHSEED='0'; PYTHONDONTWRITEBYTECODE='1'; PYTHONNOUSERSITE='1'
  PYTHONBREAKPOINT='0'; CUBLAS_WORKSPACE_CONFIG=':4096:8'
  CUDA_DEVICE_ORDER='PCI_BUS_ID'; OMP_NUM_THREADS='1'; OMP_DYNAMIC='FALSE'
  MKL_NUM_THREADS='1'; MKL_DYNAMIC='FALSE'; OPENBLAS_NUM_THREADS='1'
  NUMEXPR_NUM_THREADS='1'; VECLIB_MAXIMUM_THREADS='1'
  TF_ENABLE_ONEDNN_OPTS='0'; TF_FORCE_GPU_ALLOW_GROWTH='true'
  TF_DETERMINISTIC_OPS='1'; TF_CUDNN_DETERMINISTIC='1'
}
foreach ($key in @($start.EnvironmentVariables.Keys)) {
  if ($absent -contains $key -or @($fixed.Keys) -contains $key) {
    $start.EnvironmentVariables.Remove($key) # Environment entry only, not a file.
  }
}
foreach ($key in $fixed.Keys) { $start.EnvironmentVariables[$key] = $fixed[$key] }
$child = [System.Diagnostics.Process]::Start($start)
$child.WaitForExit()
if ($child.ExitCode -ne 0) { throw "U-Net child exited with code $($child.ExitCode)" }
```

Bash 4+ (the parentheses isolate all environment changes):

```bash
(
  absent=(
    PYTHONHOME PYTHONPATH PYTHONPYCACHEPREFIX PYTHONSTARTUP
    PYTHONINSPECT PYTHONUSERBASE PYTHONWARNINGS PYTHONOPTIMIZE
    PYTHONSAFEPATH PYTHONCASEOK CUDA_VISIBLE_DEVICES
    TORCH_ALLOW_TF32_CUBLAS_OVERRIDE TF_XLA_FLAGS XLA_FLAGS
    KMP_AFFINITY KMP_BLOCKTIME KMP_DUPLICATE_LIB_OK KMP_SETTINGS
    ATEN_CPU_CAPABILITY NPY_DISABLE_CPU_FEATURES MKL_CBWR
    MKL_ENABLE_INSTRUCTIONS CUDA_DEVICE_MAX_CONNECTIONS
    TORCH_CUDNN_V8_API_DISABLED CUBLASLT_WORKSPACE_SIZE
    TF_USE_LEGACY_KERAS NVIDIA_TF32_OVERRIDE
  )
  fixed=(
    PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
    PYTHONBREAKPOINT=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
    CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE
    MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
    TF_ENABLE_ONEDNN_OPTS=0 TF_FORCE_GPU_ALLOW_GROWTH=true
    TF_DETERMINISTIC_OPS=1 TF_CUDNN_DETERMINISTIC=1
  )
  # Remove case variants of the recorded keys; never remove filesystem paths.
  while IFS= read -r key; do
    for item in "${absent[@]}" "${fixed[@]}"; do
      if [[ ${key^^} == "${item%%=*}" ]]; then unset "$key" || exit 1; break; fi
    done
  done < <(compgen -e)
  export "${fixed[@]}" || exit 1
  exec python -s -B -m training.train_unet --dry-run
)
```

The commands preserve other environment entries, including the external data
and source roots. B7 required `CUDA_VISIBLE_DEVICES` to be absent; do not use
this example to claim the same device selection on a different multi-GPU host.
The source/profile binding above covers only the controlled formal first batch.
The complete current CLI still lacks an equivalent full-run acceptance claim;
these instructions only prepare its startup environment.

Identical terminal weights did not make the previous M2 saved prediction fields
bitwise identical: recursive rollout amplified a nonzero discrepancy, although
all tested U-Net summary values stayed within the original tolerance. The
precise inference-profile cause is not isolated. Do not treat the training
profile as proof of an exact evaluation profile, or a CPU shape check as proof
of scientific reproducibility.

`python -B -m adapters.check --model unet --checkpoint /absolute/terminal.pt --forward`
checks strict state keys/shapes and one synthetic CPU call without training.
Validation outcomes are recorded separately. Resume support belongs in the
independent training controller and must preserve optimizer, scheduler, epoch/update,
RNG and source/data/profile bindings; it is not implemented by a state-only load.

The new controller now implements same-run epoch-boundary resume with those
states. Its explicit nonformal CPU test completed two one-step epochs both
continuously and with a fresh-process resume after epoch 1: all model, optimizer,
scheduler, RNG and scientific history fields matched exactly. This 2026-09-10
test is not a repeat of the 500-epoch experiment. See [training usage](TRAINING.md).
