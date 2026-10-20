# 实验编号与算法名称

项目代码、运行入口、实验记录与结果清单统一采用论文编号 W1、W2、W3、S1、S2、S3。编号直接标识实验，不使用另一套编号或编号转换入口。

| 编号 | 实验 | 当前结果 | 正文图号 |
| --- | --- | --- | --- |
| W1 | 流场演化预测 | [W1](../results/formal/W1_flow_prediction_20261007/) | 图2A误差曲线、图2B预测场与有符号残差 |
| W2 | 跨分辨率预测 | [W2](../results/formal/W2_cross_resolution_20261007/) | 图3A跨网格曲线、图3B N128场板 |
| W3 | 方程参数识别 | [W3](../results/formal/W3_single_trajectory_20261010/)：50组无噪声参数、250槽、750行读出 | 逐例数值及汇总表 |
| S1 | 高频支路消融 | [S1](../results/formal/S1_high_frequency_branch/) | 高频支路辅助测量 |
| S2 | 递归局部修正消融 | [S2](../results/formal/S2_recursive_local_correction/) | 局部修正辅助测量 |
| S3 | 初值分布对照 | [S3](../results/formal/S3_initial_distribution/) | 初值分布辅助图 |

补充实验编号和补充图编号分别由实验索引与图注说明。

W1的算法顺序为GIFT、GIFT-Lite、FNO-2D、FNO-3D、U-NO、U-Net、ANIE；W2使用前六种，ANIE不参加跨分辨率评价。W3使用GIFT、PDE-FIND、PDE-FIND-KC、PINN-SR、PINN-SR-KC。

W1/W2的GIFT-Lite是共同有序N300名单中的前50条，其来源模型键为`GIFT-N50`；阶段epoch不变，总更新减少。S1–S3中的GIFT-Lite使用独立N1000协议的50条训练轨迹、独立日程和验证选模；两种配置的权重不能互换。

```shell
python -m scripts.run_experiment W3 --help
python -m scripts.run_experiment S3 --help
python -m scripts.verify_published_results --experiment W1 --result-dir results/formal/W1_flow_prediction_20261007
python -m scripts.verify_published_results --experiment W2 --result-dir results/formal/W2_cross_resolution_20261007
```

当前N300/单初态的W1、W2接口见[运行说明](EXPERIMENT_EXECUTION.md)；S1–S3各自的独立N1000模型、输入和统计范围见[训练协议](TRAINING_PROTOCOL.md)。实验索引直接指向这些真实计算入口和科学结果包。

原始来源路径和文件身份保存在仓库外来源登记中。仓库仅保留当前实验身份、科学测量及可核验的来源记录引用，见[来源与完整性说明](SOURCE_PROVENANCE.md)。
