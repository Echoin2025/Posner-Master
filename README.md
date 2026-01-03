# Posner注意线索范式实验项目

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PsychoPy](https://img.shields.io/badge/PsychoPy-2023.2.2-green.svg)](https://www.psychopy.org/)
[![License](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)

基于PsychoPy实现的经典Posner注意线索范式实验，包含完整的数据收集、分析和可视化流程。

## 📋 目录

- [项目简介](#项目简介)
- [实验原理](#实验原理)
- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [数据分析](#数据分析)
- [结果示例](#结果示例)
- [参考文献](#参考文献)

## 🎯 项目简介

本项目复现了Michael Posner (1980)提出的经典注意线索范式实验，探究空间线索对视觉注意定向的影响。实验使用PsychoPy平台实现，包含：

- ✅ 完整的实验程序（PsychoPy Builder & JavaScript）
- ✅ 自动化数据收集系统
- ✅ 60名被试的真实实验数据
- ✅ 完整的数据分析Python脚本
- ✅ 11张专业可视化图表
- ✅ LaTeX格式实验报告

**核心发现**：无效线索条件的反应时显著长于有效线索条件46.14ms（t=-4.122, p<0.001），证实了Posner注意线索效应。

## 🧠 实验原理

### Posner范式

Posner注意线索范式是认知心理学中研究空间注意的经典范式。实验逻辑：

1. **有效线索（Valid Cue）**：箭头方向与目标位置一致
   - 注意预先分配到目标位置
   - 反应时缩短 → **注意益处（Benefit）**

2. **无效线索（Invalid Cue）**：箭头方向与目标位置相反
   - 需要注意解除与转移
   - 反应时延长 → **注意代价（Cost）**

### 试次流程

```
注视点 (800ms) → 线索箭头 (200ms) → 目标圆圈 (至反应) → 反馈 (1000ms)
```

## 📁 项目结构

```
posner-master/
├── README.md                          # 本文件
├── posner.psyexp                      # PsychoPy实验文件
├── posner.js                          # JavaScript版本（在线实验）
├── posner-legacy-browsers.js         # 兼容旧浏览器版本
├── conditions.csv                     # 实验条件文件
│
├── images/                            # 刺激图片
│   ├── arrow-left.png                # 箭头线索
│   └── circle_red.png                # 红色目标圆圈
│
├── lib/                               # PsychoJS库文件
│   └── psychojs-2023.2.2.js
│
├── data/                              # 原始数据文件夹（60个CSV）
│   ├── S001_posner_2025-12-25.csv
│   ├── S002_posner_2025-12-26.csv
│   └── ... (共60个文件)
│
├── posner_analysis.py                 # 数据分析脚本 ⭐核心
├── cleaned_data.csv                   # 清洗后的数据
│
├── figures/                           # 生成的图表文件夹
│   ├── fig1_boxplot.png              # 箱线图
│   ├── fig2_barplot.png              # 柱状图
│   ├── fig3_histogram.png            # 直方图
│   ├── fig4_violin.png               # 小提琴图
│   ├── fig5_individual.png           # 个体差异图
│   ├── fig6_statistics.png           # 统计摘要
│   ├── fig7_individual_effects.png   # 个体效应排序
│   ├── fig8_trend.png                # 练习效应趋势
│   ├── fig9_effect_distribution.png  # 效应分布直方图
│   ├── fig10_direction.png           # 效应方向分类
│   └── fig11_categories.png          # 效应量分级
│
├── posner_analysis_report.txt        # 文本格式分析报告
└── posner_report.tex                 # LaTeX格式实验报告
```

## 🔧 环境配置

### 运行实验（PsychoPy）

```bash
# 安装PsychoPy
pip install psychopy==2023.2.2

# 或下载独立版本
# https://www.psychopy.org/download.html
```

### 数据分析（Python）

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install pandas numpy scipy matplotlib seaborn
```

**依赖库版本**：
- Python 3.8+
- pandas >= 1.3.0
- numpy >= 1.21.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

## 🚀 快速开始

### 1. 运行实验

#### 方法A：使用PsychoPy Builder（推荐）

```bash
# 打开PsychoPy
# File → Open → 选择 posner.psyexp
# 点击绿色运行按钮▶️
```

#### 方法B：在线版本

1. 上传 `posner.js` 和相关文件到Pavlovia或自己的服务器
2. 分享链接给被试
3. 数据自动保存到云端

### 2. 收集数据

- 实验会自动保存数据到 `data/` 文件夹
- 文件格式：`{participant}_{expName}_{date}.csv`
- 每位被试完成25个试次，约5-10分钟

### 3. 分析数据

```bash
# 确保data文件夹中有CSV文件
python posner_analysis.py
```

**输出**：
- ✅ 11张PNG图表（保存在 `figures/` 文件夹）
- ✅ 清洗后的数据（`cleaned_data.csv`）
- ✅ 文本分析报告（`posner_analysis_report.txt`）

## 📊 数据分析

### 数据清洗

自动化清洗流程确保数据质量：

1. **移除过快反应**：RT < 200ms（预判）
2. **移除过慢反应**：RT > 3000ms（分心）
3. **移除统计异常值**：Z-score法，±3SD以外

**本项目数据保留率：96.24%**（1586/1648）

### 统计分析

```python
# 描述性统计
- 均值、标准差、中位数
- 按条件分组统计
- Posner效应量计算

# 推断性统计
- 独立样本t检验
- Cohen's d效应量
- 显著性水平检验
```

### 可视化

生成11张专业图表：

| 图表 | 类型 | 说明 |
|------|------|------|
| Fig 1 | 箱线图 | RT分布特征 |
| Fig 2 | 柱状图 | 均值对比 |
| Fig 3 | 直方图 | 分布形态 |
| Fig 4 | 小提琴图 | 密度+箱线 |
| Fig 5 | 折线图 | 个体轨迹 |
| Fig 6 | 统计摘要 | 检验结果 |
| Fig 7 | 横向柱状图 | 个体效应排序 |
| Fig 8 | 趋势图 | 练习效应 |
| Fig 9 | 直方图 | 效应分布 |
| Fig 10 | 柱状图 | 方向分类 |
| Fig 11 | 柱状图 | 效应分级 |

## 📈 结果示例

### 主要发现

```
✓ Posner效应: 46.14 ms
✓ t统计量: -4.122
✓ p值: < 0.001 (极显著)
✓ 效应量: d = 0.21 (小到中等)
✓ 正效应比例: 75% (45/60)
```
## 🔬 实验设计

| 要素 | 描述 |
|------|------|
| **设计类型** | 单因素被试内设计 |
| **自变量** | 线索有效性（有效 vs 无效） |
| **因变量** | 反应时（毫秒） |
| **被试** | 60名（视力正常） |
| **试次** | 每人25个（随机顺序） |
| **SOA** | 200ms（线索-目标间隔） |
| **刺激** | 箭头线索 + 红色圆圈目标 |

## 📚 参考文献

1. Posner, M. I. (1980). Orienting of attention. *Quarterly Journal of Experimental Psychology, 32*(1), 3-25.

2. Posner, M. I., & Cohen, Y. (1984). Components of visual orienting. In H. Bouma & D. G. Bouwhuis (Eds.), *Attention and Performance X: Control of Language Processes* (pp. 531-556). Hillsdale, NJ: Lawrence Erlbaum Associates.

3. Posner, M. I., & Petersen, S. E. (1990). The attention system of the human brain. *Annual Review of Neuroscience, 13*(1), 25-42.

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- 感谢所有参与实验的被试
- 感谢PsychoPy开发团队提供优秀的开源工具
- 感谢Python科学计算社区（NumPy, pandas, matplotlib, seaborn, scipy）

## 📮 联系方式

如有问题或建议，欢迎联系：

- 项目主页：[(https://github.com/Echoin2025/Posner-Master/)]
- 邮箱：[Echoin2023@outlook.com]
- 课程：脑与认知科学 | 2025年春季学期

---

**⭐ 如果这个项目对您有帮助，请给我们一个Star！**

*最后更新：2025年1月*
