# Brand Insight AI — 通用消费者数据分析工作台

> **AI-Native 数据分析工具** | 上传任意 CSV → 自动分析 → 一键生成报告
>
> 使用 Claude Agent 完成全流程代码生成，构建可复用的数据分析工作流
>
> 🚀 **在线演示**: https://brand-insight-analysis-qbc3lcsznbqjbfgfdnzbjc.streamlit.app/

---

## 项目概述

这是一个**通用的消费者数据分析工作台**。上传任意 CSV 格式数据，引擎自动识别列类型、匹配分析方法、生成可视化图表和分析报告。项目采用 **"人（分析框架）× AI（代码执行）"** 的协作范式构建。

**核心价值**：不是做了一次分析，而是**构建了一个可无限复用的分析能力**。

## 使用方式

### Web 界面（推荐）
```bash
streamlit run app.py
```
浏览器打开后，拖拽 CSV 文件 → 点击"Run Analysis" → 查看结果 → 下载报告。

### 命令行
```bash
python auto_analyzer.py data/your_file.csv output/your_analysis
```
一句话生成完整分析报告。

### Python API
```python
from auto_analyzer import AutoAnalyzer
from report_generator import generate_report

analyzer = AutoAnalyzer('data/my_data.csv', output_dir='output/demo')
results = analyzer.run_all()
generate_report(results, 'output/demo/report.md')
```

## 自动分析能力

| 分析模块 | 触发条件 | 输出 |
|----------|----------|------|
| **列类型识别** | 始终执行 | 数值/分类/日期/ID/文本 自动分类 |
| **描述统计** | ≥1 数值列 | 均值/标准差/分位数/偏度/峰度 |
| **单变量可视化** | ≥1 数值或分类列 | 直方图+箱线图 / 分类柱状图 |
| **相关性分析** | ≥2 数值列 | 相关性矩阵 + 热力图 + 高相关对 |
| **分组对比检验** | ≥1 数值 + ≥1 分类列 | t检验(2组) / ANOVA(3+组) + 箱线图 |
| **客户聚类** | ≥2 数值列 | K-Means + 肘部法则 + PCA 可视化 |
| **报告生成** | 始终执行 | 结构化 Markdown 报告 |

## 项目结构

```
brand-insight-analysis/
├── README.md
├── requirements.txt
├── app.py                    # Streamlit Web 界面
├── auto_analyzer.py          # 核心分析引擎（CLI + API）
├── report_generator.py       # 报告自动生成器
├── data/
│   └── shopping_behavior.csv # 示例数据集
├── output/                   # 分析输出
├── report/
│   └── analysis_report.md    # 示例分析报告
└── 简历文案_数据分析项目.md   # 简历版本文案
```

## 技术栈

`Python` `Pandas` `NumPy` `Matplotlib` `Seaborn` `Scikit-learn` `SciPy` `Streamlit` `K-Means` `PCA` `t检验` `ANOVA`

## 验证数据集

| 数据集 | 行数 | 列数 | 来源 |
|--------|------|------|------|
| Customer Shopping Behavior | 3,900 | 18 | Kaggle |
| Cloud-Enabled Marketing Strategy | 5,000 | 27 | Kaggle |

## AI Agent 协作说明

本项目全程使用 **Claude Agent** 作为开发引擎：

| 环节 | Claude 角色 | 我的角色 |
|------|-------------|----------|
| 架构设计 | — | 定义分析引擎的功能模块和触发规则 |
| 代码实现 | 生成全部 Python 代码 | 审核逻辑、修复 bug、迭代优化 |
| 界面开发 | 生成 Streamlit 界面代码 | 设计交互流程和布局 |
| 测试验证 | 辅助排错 | 用多数据集验证通用性 |

---

*学年论文配套实践项目 | 展示营销数据分析能力 × AI工具应用能力 × 产品化思维*
