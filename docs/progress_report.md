# PV Project Progress Report

这个文件用于展示当前项目每个阶段已经完成的成果，适合给老师检查进度时使用。

## 1. 数据样本准备

当前已经准备了两类数据。

### 模拟数据

- 文件：`data/simulated_pv_data.csv`
- 行数：8,760 行
- 含义：模拟一年逐小时光伏运行数据，用于保证项目在没有真实数据时也能完整运行。
- 字段：`irradiance`、`ambient_temperature`、`module_temperature`、`humidity`、`wind_speed`、`hour`、`day_of_year`、`power_output`

### 真实数据样本

- 文件：`data/real_samples/pvdaq_system_10_2023_01_01.csv`
- 行数：1,440 行
- 数据来源：NREL / DOE PVDAQ public dataset
- 站点：PVDAQ system 10, Golden, Colorado, USA
- 时间范围：2023-01-01，一天分钟级数据
- 主要字段：`measured_on`、`ac_power__423`、`dc_power__422`、`ambient_temp__428`、`module_temp_1__429`、`poa_irradiance__421`

## 2. 数据预处理

- 脚本：`src/preprocess_data.py`
- 输入：`data/simulated_pv_data.csv`
- 输出：
  - `results/missing_values.csv`
  - `data/clean_dataset.csv`
  - `results/data_distribution.png`

当前清洗结果：

- 原始数据：8,760 行
- 清洗后数据：8,382 行
- 删除行数：378 行
- 缺失值：核心字段缺失值数量为 0

处理内容包括：

- 检查缺失值。
- 删除缺失值。
- 删除不符合物理意义的值。
- 使用 IQR 方法删除极端统计异常值。
- 可视化清洗前后的数据分布。

## 3. 构建不同天气数据集

- 脚本：`src/create_weather_datasets.py`
- 输入：`data/clean_dataset.csv`
- 输出：
  - `data/weather_datasets/all_weather_dataset.csv`
  - `data/weather_datasets/sunny_dataset.csv`
  - `data/weather_datasets/cloudy_dataset.csv`
  - `results/weather_dataset_summary.csv`
  - `results/weather_dataset_distribution.png`

划分方法：

- 先筛选白天且有发电的记录。
- 按同一小时内的辐照度分位数划分天气条件。
- 同小时辐照度较高的数据标记为 `sunny`。
- 同小时辐照度较低的数据标记为 `cloudy`。
- 中间部分标记为 `moderate`。

当前天气数据集统计：

```text
sunny:  1,263 rows, average irradiance 560.679, average power output 51.514
cloudy: 1,263 rows, average irradiance 275.080, average power output 26.340
```

这一步让项目更接近真实研究，因为晴天和阴天的发电规律不同，模型表现也可能不同。

## 4. 多数据集、多模型训练

- 脚本：`src/train_model.py`
- 数据集：
  - `all_weather`
  - `sunny`
  - `cloudy`
- 模型：
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - KNN Regressor
  - SVR
  - Decision Tree
  - Random Forest
  - Extra Trees
  - Gradient Boosting
- 输出：
  - `results/model_metrics.csv`
  - `results/best_models_by_dataset.csv`
  - `results/model_comparison.png`
  - `results/prediction_results.csv`
  - `results/predicted_vs_actual.png`

## 5. 当前模型比较结果

当前结果显示，不同天气数据集上的最佳模型不同：

```text
all_weather: Gradient Boosting, RMSE 1.1550, R2 0.9973
sunny:       Random Forest,     RMSE 1.5607, R2 0.9927
cloudy:      Lasso Regression,  RMSE 1.6843, R2 0.9738
```

这说明项目已经从“一个数据集 + 一个模型”升级为“多个天气场景 + 多个模型比较”。

## 6. 展示时可以怎么讲

可以这样讲：

> 我目前已经完成了光伏数据样本准备和数据预处理，并且进一步把数据划分成晴天和阴天两类场景。这样做是因为晴天和阴天的辐照度、组件温度和发电功率关系不同，单一模型不一定在所有天气下都最好。现在我已经在 all_weather、sunny、cloudy 三个数据集上分别训练了多个模型，并用 MAE、RMSE、R2 比较效果。初步结果显示，不同天气条件下最佳模型不同，这说明后续可以针对天气类型建立更细分的预测模型。

## 7. 下一步计划

1. 把 PVDAQ 真实数据字段映射到项目统一字段。
2. 寻找或补充真实数据中的湿度、风速字段。
3. 使用真实数据训练初版模型。
4. 加入更多天气类型，例如多云、雨天或高温天气。
5. 分析不同天气下预测误差是否可以用于异常检测。
