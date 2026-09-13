# 可续算 L-BFGS-B 状态驱动：只验证了解析二次函数

`training/pinn_lbfgsb.py` 是第一方状态/调用协议，不是新的优化算法。算法本体直接调用用户已安装 **SciPy1.7.3** 的 `_lbfgsb.setulb` 编译扩展；未复制 SciPy Python 包装器、Fortran 源码或其他 vendor 实现。它没有导入 TensorFlow，不调用 PINN，也没有改动 `adapters/pinn_runtime.py`。

## 接口核查和数值合同

已经核对 [SciPy1.7.3 的官方 Python 接口](https://github.com/scipy/scipy/blob/v1.7.3/scipy/optimize/lbfgsb.py)、已安装 f2py 的真实函数签名，以及 [TF1.15 官方 external_optimizer](https://github.com/tensorflow/tensorflow/blob/590d6eef7e91a6a7392c8ffffb7b58f2e0c8bc6b/tensorflow/contrib/opt/python/training/external_optimizer.py)。后者使用 SciPy `minimize` 的解析梯度接口并提升梯度至 float64；本驱动按该类型合同保存参数、f/g及工作区。

默认保持原锁定设置：`maxcor=50`、`maxls=50`、`maxiter=maxfun=10000`、`gtol=1e-5`、`ftol=0.1*float64_epsilon`（传给内核的 factr=0.1）。仅无边界约束、完整解析梯度；不支持有限差分、其他 SciPy 版本、任意约束或随机目标。显式 option 覆盖会进入身份绑定，测试中的短预算不是正式 PINN 预算。

版本/续算绑定包括 Python、SciPy、NumPy、处理器/平台、线程设置、已安装 `setulb` 二进制与 SciPy 接口文件 SHA、驱动源码 SHA、维度、原始 x SHA，以及调用者给出的目标/数据/参数排列身份。固定测试环境为 Python3.8.20、SciPy1.7.3、NumPy1.21.6，OMP/MKL/OpenBLAS均1线程。环境或源码不一致时拒绝恢复；不会退回另一版本的 private API。

## 恢复边界

| 返回状态 | 下一次 advance 的职责 | 持久化内容 |
|---|---|---|
| FG | 在当前 x 请求目标 f/g，再交回原内核 | 完整工作区、当前 x、上次 f/g及目标缓存，不能重新 START |
| NEW_X | 已接收一轮新迭代；继续内核或按原预算停止 | 迭代/求值/内核调用计数、待发送停止标记及全部状态 |
| 收敛、停止、错误终点 | 保留真实终止任务，不再调用内核 | 原终止任务与最终状态；终止不自动等于科学验收通过 |

数组完整保存 `x/g/f`、`l/u/nbd`、`wa/iwa`、`task/csave/lsave/isave/dsave`，另保存同一点目标缓存以避免重复求值改变计数。没有重建或近似 L-BFGS 历史，也不把参数 x 单独保存称为断点续算。工作区含计时字段；独立运行的所有工作区字节不必彼此相同，测试要求的是同一快照恢复时全数组原样，以及完整优化轨迹和终点一致。

`Journal` 使用 `allow_pickle=False` 的 NPZ和JSON、同attempt UUID、SHA完整性检查、只创建的状态目录、短非阻塞writer锁和原子LATEST提交。旧writer、不同身份、重复保存同一内核边界或损坏NPZ会拒绝，旧文件不删除。若断电发生在尚未提交的写入中，孤立目录会保留，同名目录不会被覆盖；目前未做真实断电/写盘故障注入，也未实现孤立目录的自动协调恢复。只承诺从有效已提交边界加载，不承诺从Fortran调用内部任意指令恢复。

## 最小使用方式

从源码根，在独立已安装PINN科学环境中使用，无需安装主Torch wheel：

```python
from training.pinn_lbfgsb import LBFGSB, Journal

solver = LBFGSB(x0, identity={"objective_sha256": "...", "data_sha256": "...",
                             "parameter_order": "explicit ordered parameter names"})
journal = Journal(new_output_directory, solver)
while not solver.done:
    event = solver.advance(analytic_objective)
    journal.save()  # 可选择较低检查点频率；必须处于返回边界
```

恢复时用同样x0、identity与options新建solver，再 `Journal(same_directory, solver, resume=True)`；随后继续advance，不重新初始化内核历史。目标函数接收x的副本，返回有限标量f和与x同长度的完整梯度。恢复后的第一条FG请求必须用同一个确定性目标处理；模型、数据、mask及分块方式由调用者同步绑定，当前 [KC 集成层](PINN_PHASE_SCHEDULER.md)提供该绑定，不在本工程原语中隐式寻找。

## 实际执行结果与边界

首次运行即真实通过 **4 个 unittest，2.912秒，rc0**，没有数值修正：

- 对角与耦合两种4维正定解析二次函数，对官方 `scipy.optimize.minimize` 的终点x/f、所有实际FG求值点/f/g、NEW_X迭代轨迹、计数和终止任务严格相同；分别13轮/14次求值、10轮/11次求值。
- 每种函数分别在第三次FG请求和第二次NEW_X保存；启动**4个全新Python进程**恢复。恢复前全部数组和metadata逐字节/值一致；拼接后的全部内核事件、求值轨迹、接受点、终点及计数与连续运行严格相同，没有重新START。
- maxiter=2与maxfun=1的短预算测试，终点/轨迹/停止与官方API一致；保留“完成当前接受迭代后才检查maxfun”的原语义。
- 重复提交、错误目标身份、旧writer和专门负例中损坏的NPZ均被拒绝。负例只修改测试自己生成的文件，未修改科学结果。

测试：`python -s -B -m unittest discover -s tests -p test_pinn_lbfgsb.py -v`。输出目录由 `GIFT_LBFGSB_TEST_OUTPUT` 指定；不指定时生成临时目录并保留，不批量删除。非锁定环境会明确skip，不计通过。

**本页组件测试的范围**仅为上述两个解析目标，不包含 PINN 图或完整训练。后续 [KC 集成验证](PINN_PHASE_SCHEDULER.md)已另行覆盖固定小预算 PINN 目标、TF 参数往返与全阶段联合提交，并完成首个 L-BFGS FG 边界的新进程恢复和真实 CLI 终态导出。完整非凸轨迹、正式预算、其他中断/写盘故障边界及 GPU 尚未验收；noise001/KC 完整预算任务正在 CPU 运行。此前 L-BFGS 后漂移和 open 梯度微差并未因此证明解决，open 训练继续拒绝。M1 的已训练数值终态快读是另一条路径，不运行本优化器。
