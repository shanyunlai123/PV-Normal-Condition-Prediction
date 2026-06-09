# PV Normal Condition Prediction - Progress Package

这个仓库用于展示“正常状态下光伏发电功率预测”项目的阶段性进度。当前版本已经从简单 baseline 升级为：数据预处理、真实数据样本、晴天/阴天数据集划分，以及不同模型在不同天气数据集上的对比实验。

## 当前进度

1. 已准备光伏数据样本。
2. 已找到真实光伏数据样本。
3. 已完成缺失值检查和异常值删除。
4. 已完成数据分布可视化。
5. 已输出清洗后的 `clean_dataset.csv`。
6. 已构建晴天、阴天和全量天气数据集。
7. 已在不同数据集上训练多个 AI 回归模型并比较结果。

## 仓库结构

```text
PV/
  data/
    simulated_pv_data.csv
    clean_dataset.csv
    weather_datasets/
      all_weather_dataset.csv
      sunny_dataset.csv
      cloudy_dataset.csv
    real_samples/
      README.md
      pvdaq_system_10_2023_01_01.csv
      pvdaq_systems_20250729.csv
  docs/
    progress_report.md
  results/
    missing_values.csv
    data_distribution.png
    weather_dataset_summary.csv
    weather_dataset_distribution.png
    model_metrics.csv
    best_models_by_dataset.csv
    model_comparison.png
    predicted_vs_actual.png
    prediction_results.csv
  src/
    preprocess_data.py
    create_weather_datasets.py
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
6. 晴天/阴天数据集划分代码：`src/create_weather_datasets.py`
7. 晴天数据集：`data/weather_datasets/sunny_dataset.csv`
8. 阴天数据集：`data/weather_datasets/cloudy_dataset.csv`
9. 天气数据集统计：`results/weather_dataset_summary.csv`
10. 天气数据集分布图：`results/weather_dataset_distribution.png`
11. 多模型训练代码：`src/train_model.py`
12. 模型指标对比：`results/model_metrics.csv`
13. 各数据集最佳模型：`results/best_models_by_dataset.csv`
14. 模型比较图：`results/model_comparison.png`
15. 预测效果图：`results/predicted_vs_actual.png`

## 晴天和阴天数据集怎么划分

`src/create_weather_datasets.py` 会从清洗后的数据中筛选白天发电记录，然后按同一小时内的辐照度分位数划分天气条件。

这样做的原因是：早上和傍晚的辐照度天然较低，不能简单用一个固定辐照度阈值判断阴天。因此脚本会把同一小时的数据放在一起比较：

- 辐照度处于同小时较高区间：`sunny`
- 辐照度处于同小时较低区间：`cloudy`
- 中间部分：`moderate`
- 夜间或低发电数据：`night_or_low_power`

当前数据集统计：

```text
sunny:  1,263 rows, average irradiance 560.679
cloudy: 1,263 rows, average irradiance 275.080
```

## 训练了哪些模型

当前 `src/train_model.py` 会在 `all_weather`、`sunny`、`cloudy` 三个数据集上分别训练并比较这些模型：

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

## 当前最佳结果

当前结果显示，不同天气数据集上的最佳模型并不完全相同：

```text
all_weather: Gradient Boosting, RMSE 1.1550, R2 0.9973
sunny:       Random Forest,     RMSE 1.5607, R2 0.9927
cloudy:      Lasso Regression,  RMSE 1.6843, R2 0.9738
```

这说明项目已经不只是简单训练一个模型，而是在比较不同天气场景下模型表现的差异。

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

创建晴天/阴天数据集：

```bash
python src/create_weather_datasets.py
```

运行多数据集、多模型训练和比较：

```bash
python src/train_model.py
```

