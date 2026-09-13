# PINN STRidge：外置回归接口与验证边界

`adapters.pinn_stridge` 只提供第一方加载、输入/来源绑定及有限兼容接口。使用 **Python3.8 / NumPy1.21.6**；不导入TensorFlow或Torch，不创建网络、不更新网络、不实现完整L-BFGS-B/ADO训练。两种回归方法的函数体始终从用户外置固定源码读取到内存，不复制进本仓库、数据包或生成缓存。

## 外置源与调用

设置 `GIFT_EXTERNAL_ROOT` 为外部clone的父目录，其 `pinn_sr/` 使用 [EQDiscovery](https://github.com/isds-neu/EQDiscovery/tree/9a20ebe6783e00bc53e3cdd1fd7ff07d2170f496)，commit `9a20ebe6783e00bc53e3cdd1fd7ff07d2170f496`。目标文件为 `Examples/Discovery with Single Dataset/Vorticity/NS_Vorticity_UniformSetting_OneAdam_PreADOPt.py`，canonical SHA-256为 `C5AE04D1C9E220B912C5D0ABD89D03216B85649C08236F5B9A861F98198BA261`。同时核对registry、实际HEAD、源文件hash和编译前实际字节；不自动下载、不退回未知源码，不执行上游顶层训练代码。

```python
from adapters.pinn_stridge import fit_stridge

result = fit_stridge(
    matrix=phi, target=omega_t, inherited=current_coefficients,
    l0_penalty=saved_l0, iteration=round_index,
)
# result["coefficients"] 为float64一维数组；由调用者在TF边界转float32列。
# 保留 result["l0_penalty"]，供下一ADO轮与同次任务恢复使用。
```

`phi` 为有限float32二维数组（open90列/KC4列），`omega_t` 为同排数float32向量/列，继承系数为4/90维float32或float64。真实训练应由阶段控制器核对并按原顺序收集完整物理行及输入来源；本函数允许小测试矩阵，**不会单凭shape宣称完整训练数据资格通过**。零/非有限训练列范数明确拒绝，不自动加epsilon、删除列或改归一化。

返回 `coefficients`、`l0_penalty`、`best_tolerance`、`best_error`、`accepted_history`、`source_binding`、`input_binding`、`runtime`、`audit`。数组hash同时记录shape/dtype；绑定包括原矩阵、target、继承向量、排序后的训练行、轮次、输入l0，以及外部源/registry/本适配器SHA。系数为NumPy数组，写JSON时由调用者显式 `.tolist()`。输入前后hash必须一致。

## 四个不可省略的兼容边界

1. **原行排序：**独立MT19937 seed0选80%行，选中集合不变，但按原行号排序后回归，匹配正式适配器的布尔mask语义；原上游保留随机排列。局部NumPy门面隔离seed/choice，不改变全局RNG。
2. **float64继承：**向原生会话读值接口提供本轮固定系数的float64副本，匹配正式外层初始holdout MSE/l0的提升精度。同一轮100次阈值尝试始终继承该向量，不用上一次候选替代。
3. **`lstsq(rcond=-1)`：**通过局部linalg接口显式保留正式定义，不沿用不同NumPy版本的默认值。每列L2、归一化complex64矩阵、ridge=1e-5、inner=10、tol初值/步长1、100次搜索及原生接受规则保持。
4. **精确AST命中：**仅选择 `TrainSTRidge`、`STRidge` 两方法；严格要求一个 `biginds != []` 节点，将空数组兼容判断替换为长度非0。不是整体字符串替换，也不重新实现阈值搜索或稀疏回归。

第0轮可传 `l0_penalty=None`，从其继承系数holdout MSE得到固定l0。第1–5轮必须传入持久值，不能每轮重新标定；阶段journal还须核对该值确实属于同一次任务，本函数不伪造这种跨调用证明。

**日志仅含初始项＋被接受候选，不是全部100次trial。** 上游接受日志中的tol记录发生在步长递增之后，因此输出键明确命名 `reported_tolerance_after_accept_increment`；选择的真实最优tol另见 `best_tolerance`。不可将此日志补名为正式完整拒绝/接受轨迹。

## ADO衔接与尚未完成部分

每轮按原物理行顺序收集当前图的Phi/omega_t，STRidge后立即将系数写回，再做1000次**网络＋系数**联合NAdam，重复六轮。ADO期间mask保持全1；第六轮1000步NAdam完成后读取当前系数，以精确非0建立最终mask，之后20000步post。不是第六次STRidge后立即冻结，也没有额外第七次STRidge。[L-BFGS-B 自身恢复](PINN_LBFGSB.md)已有解析目标测试；[完整阶段调度](PINN_PHASE_SCHEDULER.md)另已通过 KC mock、小预算真实联动、新进程续算及 CLI 终态导出。Known/KC 完整预算 noise001 任务正在运行、尚未验收；open 训练门禁保持关闭，不能由本组件的合成测试推定完整训练通过。

2026-09-10在Python3.8.20/NumPy1.21.6、CPU且BLAS线程1下，三个合成案例与只读正式回归oracle比较通过：128×4、192×90、持久l0/下一轮继承向量；全部最终系数、固定l0、最优tol与目标逐位一致，全局NumPy RNG不变。同轮100次继承向量一致；后轮缺l0被拒绝。此证据不是84000真实物理行、六轮ADO或完整PINN训练验证；open已有非零/chunk梯度差异仍未解决。

运行可选测试：`python -B -m unittest tests.test_pinn_stridge -v`。需显式配置 `GIFT_EXTERNAL_ROOT` 和只读 `GIFT_LEGACY_PROJECT_ROOT`；未配置或不在规定环境时明确skip，不能计作通过。可设置新的仓库外 `GIFT_PINN_STRIDGE_TEST_OUTPUT` 保存报告，不覆盖旧目录。正常运行不访问作者的正式项目；只有维护者opt-in测试读取oracle。

Third-party algorithms remain externally supplied. The adapter checks pinned source identity and applies only the documented ordering/precision/compatibility boundaries. Its accepted-only log and synthetic equivalence tests do not certify full PINN training, ADO continuation, or redistribution rights. The unresolved open-library gradient discrepancy remains unchanged.
