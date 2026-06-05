# PV Normal Condition Prediction - Progress Package

这个仓库用于展示“正常状态下光伏发电功率预测”项目的阶段性进度。当前版本重点展示：数据样本、真实数据来源、数据预处理成果，以及多个 AI 回归模型的初步训练和对比。

## 当前进度

1. 已准备光伏数据样本。
2. 已找到真实光伏数据样本。
3. 已完成缺失值检查。
4. 已完成异常值删除。
5. 已完成数据分布可视化。
6. 已输出清洗后的 `clean_dataset.csv`。
7. 已添加多个 AI 模型进行训练和比较。

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
    model_comparison.png
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
7. 多模型训练代码：`src/train_model.py`
8. 模型指标对比：`results/model_metrics.csv`
9. 模型比较图：`results/model_comparison.png`
10. 预测效果图：`results/predicted_vs_actual.png`

## 训练了哪些模型

当前 `src/train_model.py` 会训练并比较这些模型：

- Linear Regression
- Ridge Regression
- Lasso Regression
- KNN Regressor
- SVR
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting

比较指标包括：

- `MAE`: 平均绝对误差，越小越好。
- `RMSE`: 均方根误差，越小越好。
- `R2`: 决定系数，越接近 1 越好。

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

运行多模型训练和比较：

```bash
python src/train_model.py
```

## 汇报话术

可以这样说：

> 我目前已经完成了光伏数据样本准备和数据预处理。项目中保留了一个模拟数据样本用于流程验证，同时也找到了 NREL / DOE PVDAQ 的真实光伏数据样本。预处理阶段已经完成缺失值检查、异常值删除和数据分布可视化，并输出了 clean_dataset.csv。现在模型训练部分已经从两个 baseline 扩展到多个回归模型，包括线性模型、树模型、集成模型、KNN 和 SVR。下一步我会分析不同模型的误差表现，并把真实数据字段统一后接入训练流程。
