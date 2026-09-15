<div align="center">

# GIFT

**Learning interpretable field generators**  
**从观测数据学习可解释的场生成器**

</div>

![Cross-resolution prediction errors / 跨分辨率预测误差](results/formal/M3_cross_resolution/figures/cross_resolution_mean_relative_l2.svg)

<p align="center">Cross-resolution evaluation / 跨分辨率评价。Training protocols and evaluation boundaries differ by method; see <a href="docs/RESULTS.md">result notes / 结果说明</a>.</p>

## 中文

GIFT 将可解释的生成元、高频学习支路和递归局部修正结合起来，研究二维流动中的方程参数识别、递归预测与跨分辨率预测。

每个模型和实验分别运行，GIFT 的 GPU 训练默认使用加速实现并支持完整状态恢复。数据与第三方实现位于项目外；第三方代码的固定来源和必要调整见[外部适配说明](docs/EXTERNAL_ADAPTATIONS.md)。数据通过 `GIFT_DATA_ROOT` 指定，外部源码通过 `GIFT_EXTERNAL_ROOT` 指定。

[实验报告](EXPERIMENTS.md) · [安装与独立运行](docs/SETUP.md) · [训练协议](docs/TRAINING_PROTOCOL.md) · [计算效率](docs/PERFORMANCE.md) · [实验图像](docs/FIGURES.md)

项目包含六项实验的实测汇总、五幅 SVG 和完整模型权重。预测图表展示 GIFT（1000 条训练轨迹）、GIFT-Lite（50 条训练轨迹）及基线在 180 条独立测试轨迹上的结果。可直接阅读结果、用权重快速重算，或按模型分别从零训练与断点续算。数据包单独提供，不包含在 GitHub 仓库中。

结论及适用范围见[结果说明](docs/RESULTS.md)：相同 epoch 不代表相同计算量，固定种子也不保证跨设备逐比特一致。

## English

GIFT combines an interpretable generator, a learned high-frequency branch and local correction for equation identification and flow prediction.

Models and experiments have independent entry points. GIFT uses accelerated GPU training by default with complete-state recovery. Data and third-party implementations are supplied separately. See the [external adaptation guide](docs/EXTERNAL_ADAPTATIONS.md) for source boundaries and explicit adjustments.

[Experiment report (Chinese)](EXPERIMENTS.md) · [Setup and execution](docs/SETUP.md) · [Training protocol](docs/TRAINING_PROTOCOL.md) · [Performance](docs/PERFORMANCE.md) · [Experiment figures](docs/FIGURES.md)

The project includes measured summaries for six experiments, five SVG figures and complete model weights. Prediction figures show GIFT (1,000 training trajectories), GIFT-Lite (50 training trajectories) and the baselines on 180 independent test trajectories. Read the results, recompute them with the supplied weights, or train and resume each model independently. The data package is supplied separately, outside GitHub.

See the [result notes](docs/RESULTS.md) for findings and evaluation boundaries. Equal epochs do not imply equal compute, and fixed seeds do not guarantee cross-device bitwise equality.

## Acknowledgements / 致谢

We thank the authors and contributors of the external projects listed in `external_sources.json`. Their research and shared implementations support these comparisons. Please cite the corresponding work and retain attribution.

感谢所引用仓库的作者与贡献者。使用相关方法时，请引用原始研究并保留署名。

## License / 许可

GIFT software and documentation use the [MIT license](LICENSE). Data and external projects have their own terms.
