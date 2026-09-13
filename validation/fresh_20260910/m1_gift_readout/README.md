# 新训练低频 GIFT 模型的 M1 参数读出

本目录记录三个新完成正式预算训练的低频模型，在CPU上独立读取nu/beta/gamma参数的结果。**9参数、36项原表数值比较全部满足原容差：9项精确相同、0项超差。** 不是完整M1从零验收，也不是全部数值逐比特相同。

读取过程不重新训练，但会执行GIFT分量forward和原P21投影。仅加载本次新完成的三个终态权重，不加载旧发布权重，不使用GPU。模型从零训练/预算验证由旁边的 `gift_low_noise_000`、`gift_low_noise_001`、`gift_low_noise_010` 证据目录分别记录；本目录不重新作出训练完成断言，也不修改model catalog。

| 条件 | 实际新终态文件 | 比较项 | 精确项 | 超差 |
| --- | --- | ---: | ---: | ---: |
| noise_000 | gift_main.pt | 12 | 3 | 0 |
| noise_001 | gift_noise_001.pt | 12 | 3 | 0 |
| noise_010 | gift_noise_010.pt | 12 | 3 | 0 |

比较列为estimate、reference、absolute_error和relative_error_percent。reference真值仅在读出后用于计算误差，不传入算法。保持原 `abs(candidate-published) <= 0.0005 + 0.05*abs(published)`。没有替换原表、放宽容差或筛去非精确项。

三个独立进程均rc0，墙钟约7.297、8.234、7.641秒；包括导入和完整输入SHA检查。数值环境与先前CPU参数读出profile完全一致：Python3.10.19、NumPy2.2.6、SciPy1.15.3、Torch2.10.0+cu126，CPU1线程、CUDA_VISIBLE_DEVICES=-1，无TF32全局覆盖。Torch编译带CUDA不表示本次使用了GPU。

`comparison.json` 保留全部36项实际比较和每个执行identity、新checkpoint/原训练验证/原读出result/commit哈希。它是规范化JSON重序列化，已逐值核验与原外部比较相同；原文件SHA另在manifest中保留，不冒充字节相同。

`job_records.json` 将三个原子job的result.json、summary.csv、native_coefficients.csv、COMPLETE.json原UTF-8文本作为JSON字符串保存。全部12份嵌入文本重新编码后的字节数/SHA已核验，保留原执行来源和完成提交；不要把这个容器文件本身的SHA与嵌入原文件SHA混同。三个旁系训练verification链接及其原verification SHA、新checkpoint SHA也已核验匹配。没有加入模型源码、权重副本或本机绝对路径。

复用同一入口时显式指定某一条件自己训练完成的权重及其SHA：

```text
python -B -m experiments.formal.m1_equation_identification.run --execute --method GIFT --condition noise_001 --device cpu --output <new-output-root> --checkpoint <new-gift_noise_001.pt> --checkpoint-sha256 <actual-sha256>
```

源代码、条件数据和checkpoint绑定必须一致；每条件独立job，失败保留，不会连带重训其他模型。此证据不能与旧PINN参考系数读取拼成“完整新M1训练已通过”；新PINN完整预算及完整五方法科学断言仍须各自完成验证。
