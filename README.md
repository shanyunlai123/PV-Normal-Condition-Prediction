# PV Normal Condition Prediction - Progress Package

这个仓库目前只保留下周五进度检查需要展示的内容：数据样本、真实数据来源、数据预处理成果，以及 AI 模型刚开始训练的 baseline 结果。

## 当前进度

1. 已准备光伏数据样本。
2. 已找到真实光伏数据样本。
3. 已完成缺失值检查。
4. 已完成异常值删除。
5. 已完成数据分布可视化。
6. 已输出清洗后的 `clean_dataset.csv`。
7. 已开始 Linear Regression 和 Random Forest baseline 模型训练。

## 仓库结构

```text
PV/
  data/
    simulated_pv_data.csv
    clean_dataset.csv
    real_samples/
      README.md
      pvdaq_system_10_2023_01_01.csv
      pvdaq_systems_20250729.csv
  docs/
    progress_report.md
  results/
    missing_values.csv
    data_distribution.png
    model_metrics.csv
    predicted_vs_actual.png
    prediction_results.csv
  src/
    preprocess_data.py
    train_model.py
  README.md
  requirements.txt
```

## 每一步展示什么

详细展示材料在：

```text
docs/progress_report.md
```

可以按这个顺序给老师看：

1. 数据样本：`data/simulated_pv_data.csv`
2. 真实数据样本：`data/real_samples/pvdaq_system_10_2023_01_01.csv`
3. 缺失值检查：`results/missing_values.csv`
4. 数据预处理代码：`src/preprocess_data.py`
5. 清洗后数据：`data/clean_dataset.csv`
6. 数据分布图：`results/data_distribution.png`
7. 初步模型训练代码：`src/train_model.py`
8. 初步模型指标：`results/model_metrics.csv`
9. 预测效果图：`results/predicted_vs_actual.png`

## 真实数据来源

真实样本来自 NREL / DOE PVDAQ public dataset。

当前保留的真实样本：

```text
data/real_samples/pvdaq_system_10_2023_01_01.csv
```

该样本是 PVDAQ system 10 在 2023-01-01 的分钟级光伏运行数据，包含：

- `measured_on`
- `ac_power__423`
- `dc_power__422`
- `ambient_temp__428`
- `module_temp_1__429`
- `module_temp_2__430`
- `module_temp_3__431`
- `poa_irradiance__421`

## 如何运行当前阶段代码

安装依赖：

```bash
pip install -r requirements.txt
```

运行数据预处理：

```bash
python src/preprocess_data.py
```

运行 baseline 模型训练：

```bash
python src/train_model.py
```

## 下周五汇报话术

可以这样说：

> 我目前已经完成了光伏数据样本准备和数据预处理。项目中保留了一个模拟数据样本用于流程验证，同时也找到了 NREL / DOE PVDAQ 的真实光伏数据样本。预处理阶段已经完成缺失值检查、异常值删除和数据分布可视化，并输出了 clean_dataset.csv。AI 模型现在刚进入初步训练阶段，已经用 Linear Regression 和 Random Forest 做了 baseline，下一步会把真实数据字段进一步统一到训练流程里。
