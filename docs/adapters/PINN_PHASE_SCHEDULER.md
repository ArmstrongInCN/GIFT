# KC 阶段调度：实现与验证边界

`training/train_pinn.py` 提供第一方 `TensorFlowBackend`、`Coordinator` 和 `CoupledJournal`。它不包含外部网络、NAdam、STRidge 或 L-BFGS-B 算法实现；调用分别由 [PINN 图适配](PINN_ENVIRONMENT.md)、[外置 STRidge](PINN_STRIDGE.md)、[SciPy 状态驱动](PINN_LBFGSB.md) 提供。

**阶段调度的mock测试、CPU/GPU小预算KC全阶段联动及新进程续算已通过。noise001/known完整GPU训练也已完成，但原M1三行12个数值有2项失败，不能声明数值验收通过。** 不能将这个模块、计划输出或test-only `complete` 状态当作M1科学结果。Open调度仍拒绝；其已知非零系数诊断有160/26133梯度项不完全相同（最大2.32830644e-10），没有证明该微差造成报告级偏差；单次隔离补形`[90,1]`尝试仍失败，未集成。

新增 `--device cpu|gpu`，默认 `cpu`；GPU仅允许known/KC。GPU02另已实际通过两组tiny完整梯度/更新、tiny完整FSM/L-BFGS新进程续算、全24000/84000行单次更新三门，精确范围与SHA见 [GPU环境及证据](PINN_ENVIRONMENT.md#kc-gpu-三项有界验证与独立启动)。GPU公开CLI的tiny新任务与已完成任务复用随后也实际通过。独立noise001/known完整GPU训练现已按原预算完成（rc0，3433.60秒）；源/输入/预算/终态校验通过，数值检验却因gamma两个派生误差字段失败。原noise001 CPU任务仍在2000步有效提交后暂停保全，新进程restore-only通过；不是完成训练，也不是跨设备续算。

## 原流程保持

| 阶段 | 默认预算和设置 |
| --- | --- |
| 预 NAdam | 5000步，lr=1e-3、physics=1、L1=1e-7 |
| 预 L-BFGS-B | maxiter=maxfun=10000，maxcor=maxls=50、gtol=1e-5、ftol=0.1×float64 epsilon |
| ADO | 6轮，每轮 STRidge 后1000步联合 NAdam；lr=1e-4、physics=2、L1=0 |
| 最终 mask | 第6轮 NAdam结束后，由当前系数精确 `!=0` 生成；不是第6轮STRidge的support |
| post NAdam | 20000步、lr=1e-4、physics=2、L1=0；没有额外 post L-BFGS-B |

遵循原 Amendment002：**所有 NAdam 阶段沿用同一优化器，联合更新网络和系数**。ADO的mask一直全1，允许STRidge置零项在NAdam中重新非零；不按真值、符号或幅值补选。每轮收集当前图的完整84000个物理点（60000 LHS后接24000训练坐标），按原顺序分块32768并拼接float32 library/omega_t；L-BFGS参数/梯度接口float64，赋回图float32。首轮继承系数产生的l0固定沿用六轮。原生STRidge日志只有初始和接受项，不伪装成全100次尝试日志。

上述有界图检查与后来的noise001/KC完整GPU训练分别保留。完整训练执行了31000次NAdam和6轮STRidge，L-BFGS按原预算实际终止；**原数值比较2/12失败**，其他10项通过，不能以完成阶段替代验收。旧CPU任务仍暂停保全。修改任何预算必须显式 `diagnostic=True` / `--diagnostic`，且轮数最多6；这些结果始终test-only。L-BFGS的预算停止等实际终止任务保留，不冒充收敛，也不自动成为科学验收通过。

## 同一任务的完整状态恢复

一个提交同时保存所有TF变量（包含网络、系数、NAdam slots/beta和梯度累计量）、mask、全局NAdam步数、阶段/轮次/步数、l0及STRidge记录。活跃L-BFGS阶段同时保存其原始起点、完整reverse-communication数组、缓存、计数和任务状态，不重新START。Python和NumPy随机状态一并保存；当前图训练没有新增随机采样算子。

身份绑定覆盖源码、外部源、初始化/采样/数据、参数顺序、数值环境、预算和检查点间隔。安全NPZ禁止pickle，对NPZ/JSON检查SHA；同一UUID任务、只创建的nonce状态目录、短writer锁、原子LATEST提交。不同身份、旧writer和重复边界提交拒绝。未提交的孤立目录保留，不覆盖或删除；续算只使用有效已提交边界。

默认NAdam每1000步、L-BFGS每100个FG请求或100个NEW_X接受迭代返回边界标记应保存，另外每个阶段转换保存。没有在每次FG永久留一份约21MB工作区。中途断电最多重做最近已提交边界之后的工作；STRidge作为整次调用提交，提交前恢复可重做该次，提交后直接进入NAdam。尚未做实际断电/磁盘故障注入，不能承诺从计算内核任意指令位置恢复。

内部调用协议为：先创建新后端和 `Coordinator`，创建 `CoupledJournal` 并保存初始边界；每次 `advance()` 后若 `checkpoint_due` 则保存。恢复时先构造相同的新后端/协调器，再 `CoupledJournal(same_directory, coordinator, resume=True)` 联合加载，随后继续。CLI现已接通同一循环；默认仍只plan，必须显式 `--execute`。计划中的 `tiny_integration_validated=true` 表示小预算工程验证；`full_budget_validated=false`保留完整预算科学验收未通过的边界，不表示noise001/KC程序未走完。M1实验runner的PINN快速路径只读取历史数值终态，不调用本训练器；本次新训练及其失败比较见 [独立便携证据](../../validation/fresh_20260910/pinn_known_noise_001/README.md)。

## 当前可运行的无计算检查

在独立PINN环境中，从源码根运行，无需安装主Torch wheel：

```powershell
python -s -B -m training.train_pinn --mode known --device cpu --condition noise_001 --output ../gift_runs/pinn_kc_cpu_001
python -s -B -m unittest discover -s tests -p test_train_pinn_fsm.py -v
```

第一条仅打印计划，不读科学数据、不建图、不创建输出。第二条只用明确的假后端/假优化器验证调度，不导入TF、Torch或SciPy；由 `GIFT_PINN_FSM_TEST_OUTPUT` 指定全新外部测试输出目录，否则创建并保留系统临时目录。

实际6个mock测试涵盖：原预算/诊断门、8个阶段边界的完整状态/RNG恢复及连续轨迹exact、17个假NAdam更新与6次STRidge的正确次序和最终mask、整次STRidge提交语义、稀疏LBF保存策略、错误身份/旧writer/重复保存拒绝及孤立目录保留。它们不测试真实损失、梯度、Fortran高维线搜索或与原训练结果的数值一致性。

## 已实际完成的有界实图集成

经单独授权，固定noise001前8个训练观测、原顺序前12个LHS物理点，chunk=4；同一未训练初始化；每条2步预NAdam、LBF maxiter=2/maxfun=4、仍6轮各2步ADO、3步post。全部CPU单线程、CUDA_VISIBLE_DEVICES=-1；没有更换行、参数或数值修补。

第一条连续完成；第二条在首个LBF FG返回边界完整保存并退出，然后由**另一个Python进程**恢复完成。三个子进程实际rc0，用时13.780、10.484、13.703秒。每条逻辑任务17个NAdam更新、6次原生STRidge调用；LBF均2个接受迭代、4次目标/全梯度求值、7次内核调用，真实终点为迭代预算STOP，非收敛。

连续与续算终点的全部81个TF变量、mask、iteration、原始/有效系数、l0、六轮STRidge记录及FSM完全一致；拼接后的全部NAdam/阶段/LBF事件、LBF各求值x/f/完整梯度指纹也完全一致。恢复时全部保存的模型及LBF游标数组/schema/metadata匹配，包括工作区；不要求两条独立任务的Fortran计时字段相同。

上述CPU首次集成即通过，无耦合代码修正。这只证明固定小数据和短预算下的组件联动及单边界新进程恢复，不证明完整非凸训练轨迹、完整84000行回归、其他中断边界或此前正式结果漂移已解决。后来GPU01的4项系数梯度差通过GPU02唯一KC mask静态shape `[4,1]` 修正消除，未改精度或容差；不要把两次设备验证混为一次无修改验收。随后noise001/KC的5000+完整LBF+6000+20000正式预算已完成，但终态数值比较仍失败。该run的pre5000全部82个状态/诊断张量exact，L-BFGS后19个模型张量不同；参数/梯度打包顺序已核相同，尚未确定具体原因。M1已训练终态的快速读出不能替代本训练器的科学验收；数值与边界详情见 [复现状态](../REPRODUCIBILITY_STATUS.md)。

## 独立训练、同任务续算及终态工件

在独立PINN环境中设置 `GIFT_DATA_ROOT` 为外部数据包，`GIFT_EXTERNAL_ROOT` 为外部源码clone的父目录。CPU显式要求 `CUDA_VISIBLE_DEVICES=-1`，OMP/MKL/OpenBLAS线程均为1。可先去掉 `--execute` 检查计划；下面命令展示**当前源码新建CPU任务**及该任务未来的续算，两条不能同时运行，不是继续那条已保全的旧CPU任务：

```powershell
python -s -B -m training.train_pinn --mode known --device cpu --condition noise_001 --output ../gift_runs/pinn_kc_cpu_001 --execute
python -s -B -m training.train_pinn --mode known --device cpu --condition noise_001 --output ../gift_runs/pinn_kc_cpu_001 --resume --execute
```

GPU使用独立的新目录及 `--device gpu`，先按 [专用子进程PATH示例](PINN_ENVIRONMENT.md#kc-gpu-三项有界验证与独立启动) 提供该legacy环境自己的 `Library/bin` 与环境根目录，显式 `CUDA_VISIBLE_DEVICES=0`，保持线程1。不要全局修改PATH、套用主Torch环境或擅自改TF32。`adapters.pinn_runtime` 的仅NAdam CLI仍为CPU，不能给它传 `--device gpu`。

旧CPU任务的source/journal/launch均已保全，原记录不改。需要继续它时，必须使用归档的冻结源码、原命令和原CPU环境，从有效LATEST提交恢复；当前新源即使选择CPU也不满足旧source identity。恢复只读检查已通过，不表示恢复后又训练了一步，亦不支持CPU→GPU迁移。

条件可独立选noise_000/noise_001/noise_010。初始化固定读取 `auxiliary/pinn_initialization/network_initialization_tf115_seed1234.npz`，采样固定读取 `auxiliary/pinn_sampling/<condition>_seed1234.npz`；clean数据为 `standard_ns_n64_full_spectrum.h5`，噪声数据为 `m1_parameter_identification/<condition>.h5`。原初始化SHA、采样SHA与该条件H5的固定配对由运行时强核验；运行结束再次核验输入未变。不提供加载已训练权重的fresh参数。

输出必须位于代码/数据包/实际输入来源目录之外。Fresh拒绝已有目录；resume仅接本目录的CoupledJournal完整已提交状态，不从其他模型、其他run或已发布参数表拼接状态。已完成run的resume仅核验并复用相同终态，不再次advance。正式默认chunk32768与完整行集不可静默变更；显式修改chunk/样本数/预算必须 `--diagnostic`。

完成后只创建安全 `terminal_state.npz`（全部数值TF状态、mask、四项raw/effectivecoeff）、`result.json`（library顺序w/advection/laplacian/q、协议/预算/阶段记录/数据与源绑定）及最后的 `COMPLETED.json` 文件哈希提交。不复制TF meta graph或外部算法源码。所有结果明示 `scientific_acceptance=false`，tiny另外标记 `diagnostic_test_only=true`、`eligible_for_formal_M1=false`；完整预算工件也须独立与原参考验收，不能把“完成训练”解释为数值合格。

终态文件不覆盖：已有文件必须与当前完整状态吻合，损坏或不匹配则拒绝。最终commit以原子create-only硬链接发布（要求输出文件系统支持硬链接）；其已同步临时链接保留，不删除旧文件。若在NPZ/JSON写入中断电而留下不完整文件，目前会明确拒绝自动覆盖，须人工确认处理；不伪装成已完成。

新CLI/导出另经4个mock测试，加原6个调度测试，实际10项通过。随后仅运行一次与上述相同的tiny真实CLI：rc0，CPU约16.7秒；全部81TF变量、mask、raw/effectivecoeff和l0/STRidge/FSM相对先前连续tiny轨迹终点exact，安全NPZ及result/完成commit的SHA均核验通过，没有额外科学修正。测试工件仍不是正式M1结果。

GPU公开CLI现另有真实新任务/完成复用两次调用：分别rc0、17.756秒与10.204秒，同一tiny run的 `already_complete=false→true`，17次NAdam和6轮STRidge，终态commit正确；复用后的整个job inventory逐字节不变，没有重复advance或添加更新。它验证的是**已完成任务复用**，不同于Gate2在L-BFGS中途退出后恢复的检查。两者均不解除 `diagnostic=true` / `scientific_acceptance=false`；实际结果SHA和启动环境见 [PINN环境说明](PINN_ENVIRONMENT.md#kc-gpu-三项有界验证与独立启动)。后续已完成的正式GPU任务不是这些tiny或旧CPU任务的续算，其数值失败亦单独保留。
