# 项目简介

本项目面向大学课程项目与答辩展示，研究 Module 1（模块1）：正常运行条件下的光伏发电功率预测。项目利用天气条件、时间信息与光伏运行数据，建立机器学习回归模型，对正常状态下的 Power Output（发电功率）进行预测。

当前项目已经完成数据预处理、真实数据样本收集、天气条件分类、多模型训练、模型评估、Feature Importance（特征重要性）分析，以及 Weather Impact（天气影响）分析。项目不仅比较不同模型，也使用实际统计结果解释模型表现差异，为后续 Module 2（模块2）的异常检测提供正常运行基准。

当前项目结构如下：

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
    feature_importance.csv
    feature_importance.png
    weather_analysis.csv
    weather_comparison.png
    weather_analysis_conclusion.txt
  src/
    preprocess_data.py
    create_weather_datasets.py
    train_model.py
    feature_importance_analysis.py
    weather_impact_analysis.py
  README.md
  requirements.txt
```

详细的阶段性展示材料位于：

```text
docs/progress_report.md
```

建议课程汇报时依次展示数据样本、真实数据来源、缺失值检查、数据预处理、天气分类、多模型训练、模型评估、特征重要性、天气影响证据，以及模块1对模块2的支持。

# 项目目标

本项目的主要目标如下：

1. 建立正常运行条件下的光伏发电功率预测模型。
2. 比较不同机器学习模型在 All Weather（全天气）、Sunny（晴天）和 Cloudy（阴天）条件下的预测性能。
3. 分析 Irradiance（光照强度）、Temperature（温度）、Voltage（电压）和 Current（电流）对发电功率预测的影响。
4. 使用数据统计结果解释天气条件为什么会影响预测准确率。
5. 为模块2提供 Expected Normal Power Output（预期正常发电功率），用于识别潜在异常运行状态。

# 数据集介绍

项目同时保留模拟数据、清洗数据、天气分类数据和真实光伏数据样本。

- `data/simulated_pv_data.csv`：包含 8,760 条逐小时模拟光伏运行记录，用于验证完整项目流程。
- `data/clean_dataset.csv`：经过缺失值检查与异常值处理后的训练数据。
- `data/weather_datasets/all_weather_dataset.csv`：全天气数据集。
- `data/weather_datasets/sunny_dataset.csv`：晴天数据集。
- `data/weather_datasets/cloudy_dataset.csv`：阴天数据集。
- `data/real_samples/pvdaq_system_10_2023_01_01.csv`：真实分钟级光伏运行样本。

## 数据来源

真实数据样本来自 NREL（美国国家可再生能源实验室）与 DOE（美国能源部）的 PVDAQ（Photovoltaic Data Acquisition，光伏数据采集）公开数据集。

当前保留的真实样本为 PVDAQ system 10 在 2023-01-01 的分钟级光伏运行数据：

```text
data/real_samples/pvdaq_system_10_2023_01_01.csv
```

该真实样本包含交流功率、直流功率、环境温度、组件温度与阵列平面光照强度等字段。真实样本中的 Voltage（电压）与 Current（电流）字段尚未映射进入当前训练数据，因此当前模型无法可靠计算这两个特征的重要性。

## 数据字段说明

当前模型使用的主要字段包括：

- `irradiance`：Irradiance（光照强度），表示到达光伏组件的太阳辐射水平。
- `ambient_temperature`：Ambient Temperature（环境温度）。
- `module_temperature`：Module Temperature（组件温度）。
- `humidity`：Humidity（湿度）。
- `wind_speed`：Wind Speed（风速）。
- `hour`：一天中的小时，用于表示日内变化规律。
- `day_of_year`：一年中的第几天，用于表示季节变化规律。
- `power_output`：Power Output（发电功率），是模型需要预测的目标变量。

真实 PVDAQ 样本还包含：

- `measured_on`
- `ac_power__423`
- `dc_power__422`
- `ambient_temp__428`
- `module_temp_1__429`
- `module_temp_2__430`
- `module_temp_3__431`
- `poa_irradiance__421`

# 数据预处理

数据预处理脚本为 `src/preprocess_data.py`。该脚本执行以下步骤：

1. 读取原始 CSV 光伏数据。
2. 检查核心字段缺失值，并输出 `results/missing_values.csv`。
3. 删除不符合物理规律的异常值。
4. 使用 IQR（Interquartile Range，四分位距）方法删除极端统计异常值。
5. 生成清洗后的 `data/clean_dataset.csv`。
6. 使用 Matplotlib（Python 绘图库）生成清洗前后数据分布图 `results/data_distribution.png`。

当前数据预处理结果为：

- 原始数据：8,760 行。
- 清洗后数据：8,382 行。
- 删除异常或极端记录：378 行。
- 核心字段缺失值数量：0。

安装依赖并运行数据预处理：

```bash
pip install -r requirements.txt
```

```bash
python src/preprocess_data.py
```

# 天气条件分类

天气分类脚本为 `src/create_weather_datasets.py`。脚本首先筛选白天且存在有效发电的记录，再按照相同小时内的光照强度分位数划分天气条件。

采用同小时比较的原因是，早晨与傍晚的光照强度天然低于正午，因此不能仅使用固定光照阈值判断天气。当前分类规则如下：

- 同小时光照强度处于较高区间：`sunny`
- 同小时光照强度处于较低区间：`cloudy`
- 同小时光照强度处于中间区间：`moderate`
- 夜间或低功率记录：`night_or_low_power`

运行天气分类：

```bash
python src/create_weather_datasets.py
```

## 晴天数据集（Sunny Dataset）

Sunny Dataset（晴天数据集）包含 1,263 条记录，平均光照强度为 560.679，平均发电功率为 51.514 kW。该数据集具有较强的光照强度与发电功率关系。

```text
sunny:  1,263 rows, average irradiance 560.679
cloudy: 1,263 rows, average irradiance 275.080
```

## 阴天数据集（Cloudy Dataset）

Cloudy Dataset（阴天数据集）包含 1,263 条记录，平均光照强度为 275.080，平均发电功率为 26.340 kW。与晴天相比，阴天数据具有更高的相对光照波动，光照强度与发电功率之间的相关性也更弱，因此预测难度更高。

## 全天气数据集（All Weather Dataset）

All Weather Dataset（全天气数据集）包含 8,382 条记录，覆盖晴天、阴天、中等天气、夜间和低功率运行状态。该数据集样本数量最多，能够表示更广泛的运行范围，但其良好预测结果也需要结合较大的样本规模进行解释。

# 机器学习模型

训练脚本 `src/train_model.py` 在三个天气数据集上训练并比较以下已有回归模型：

- Linear Regression（线性回归）
- Ridge Regression（岭回归）
- Lasso Regression（套索回归）
- KNN Regressor（K近邻回归）
- SVR（支持向量回归）
- Decision Tree（决策树）
- Random Forest（随机森林）
- Extra Trees（极端随机树）
- Gradient Boosting（梯度提升树）

本项目不假设某个模型在所有天气条件下均表现最佳，而是通过已有的 MAE、RMSE 和 R² 结果选择各数据集的最佳模型。

运行多数据集、多模型训练和比较：

```bash
python src/train_model.py
```

# 模型评估指标

模型评估结果保存于 `results/model_metrics.csv`，各天气数据集的最佳模型保存于 `results/best_models_by_dataset.csv`。项目使用 MAE、RMSE 和 R² 对预测性能进行评价。

## 平均绝对误差（MAE）

MAE（Mean Absolute Error，平均绝对误差）表示预测值与真实值之间绝对误差的平均值，单位为 kW。MAE 越小，说明模型的平均预测偏差越小。

## 均方根误差（RMSE）

RMSE（Root Mean Squared Error，均方根误差）对较大的预测误差更加敏感。RMSE 越小，说明模型出现大误差的程度越低。本项目以 RMSE 最低作为选择各数据集最佳模型的主要依据。

## 决定系数（R²）

R²（Coefficient of Determination，决定系数）用于衡量模型对发电功率变化的解释能力。R² 越接近 1，说明模型能够解释的数据变化比例越高。

# 实验结果

已有实验结果表明，不同天气条件下的最佳模型不同：

```text
all_weather: Gradient Boosting, RMSE 1.1550, R2 0.9973
sunny:       Random Forest,     RMSE 1.5607, R2 0.9927
cloudy:      Lasso Regression,  RMSE 1.6843, R2 0.9738
```

天气条件最佳模型的完整指标如下：

```text
all_weather: R2 0.9973, MAE 0.7014, RMSE 1.1550
sunny:       R2 0.9927, MAE 1.1837, RMSE 1.5607
cloudy:      R2 0.9738, MAE 1.3266, RMSE 1.6843
```

## 全天气（All Weather）

All Weather（全天气）条件下，Gradient Boosting（梯度提升树）表现最佳，其 MAE 为 0.7014、RMSE 为 1.1550、R² 为 0.9973。

Gradient Boosting 能够逐步组合多个弱学习器，对广泛运行范围中的非线性关系进行细致修正。全天气数据集样本数量较多，并包含多种运行状态，这可能解释了 Gradient Boosting 的性能优势。

## 晴天（Sunny）

Sunny（晴天）条件下，Random Forest（随机森林）表现最佳，其 MAE 为 1.1837、RMSE 为 1.5607、R² 为 0.9927。

晴天数据具有较强且相对稳定的光照强度与发电功率关系。Random Forest 能够学习主要光照规律，同时捕捉温度和时间特征带来的次要非线性影响。

## 阴天（Cloudy）

Cloudy（阴天）条件下，Lasso Regression（套索回归）表现最佳，其 MAE 为 1.3266、RMSE 为 1.6843、R² 为 0.9738。

阴天数据具有较弱的光照强度与发电功率相关性，并且相对光照波动更高。Lasso Regression 的正则化机制能够限制复杂度，在规模较小、稳定性较低的数据集中降低过拟合风险。

# 特征重要性分析

Feature Importance Analysis（特征重要性分析）用于解释不同输入特征对 Power Output（发电功率）预测的影响程度。

分析方法如下：

- 树模型使用 `feature_importances_`。
- 线性模型使用标准化后的系数绝对值。
- `temperature` 由 `ambient_temperature` 和 `module_temperature` 的重要性合并得到。
- 所有输出按重要性自动排序。

当前整体最佳模型 Gradient Boosting 的特征重要性结果如下：

```text
irradiance:  99.87%
temperature:  0.13%
voltage:      not used in current model
current:      not used in current model
```

各天气数据集最佳模型的特征重要性证据如下：

| Dataset | Best model | Irradiance importance | Temperature importance | Method |
|---|---|---:|---:|---|
| all_weather | Gradient Boosting | 99.88% | 0.12% | `feature_importances_` |
| sunny | Random Forest | 99.39% | 0.61% | `feature_importances_` |
| cloudy | Lasso Regression | 93.96% | 6.04% | standardized absolute coefficients |

Irradiance（光照强度）是三个数据集最佳模型中最重要的已使用特征。这一结果符合光伏系统物理规律，因为到达光伏组件的太阳能量直接决定可转换的电能。Temperature（温度）具有次要但合理的影响，因为组件温度变化会影响光电转换效率。

阴天模型对温度的相对重要性为 6.04%，高于晴天和全天气模型。这说明阴天条件下除光照强度外，次要因素对发电功率变化的解释作用更加明显，也为阴天预测难度较高提供了论据。

Voltage（电压）和 Current（电流）尚未作为当前训练数据的输入特征，因此不能可靠计算其重要性。结果文件将它们标记为 `used_in_model=False`，而不是给出缺乏证据的解释。

输出文件包括：

- `results/feature_importance.csv`
- `results/feature_importance.png`
- `results/feature_importance_all_weather.csv`
- `results/feature_importance_sunny.csv`
- `results/feature_importance_cloudy.csv`
- `results/feature_importance_comparison.png`

运行特征重要性分析：

```bash
python src/feature_importance_analysis.py
```

```bash
python scripts/feature_importance_analysis.py
```

# 天气影响分析

Weather Impact Analysis（天气影响分析）使用已有模型指标和实际数据统计结果，解释不同天气条件对预测性能的影响。

天气条件最佳模型的性能比较如下：

- 全天气数据集整体预测最准确，最佳模型为 Gradient Boosting。
- 晴天数据集预测性能居中，最佳模型为 Random Forest。
- 阴天数据集预测最困难，最佳模型为 Lasso Regression。

三个数据集的天气统计证据如下：

| Dataset | Irradiance mean | Irradiance std | Irradiance CV | Power mean | Power std | Irradiance-power correlation |
|---|---:|---:|---:|---:|---:|---:|
| all_weather | 194.169 | 244.163 | 1.257 | 18.038 | 22.573 | 0.9973 |
| sunny | 560.679 | 214.581 | 0.383 | 51.514 | 18.737 | 0.9930 |
| cloudy | 275.080 | 112.726 | 0.410 | 26.340 | 10.621 | 0.9866 |

晴天数据的相对光照波动低于阴天数据，光照强度变异系数分别为 0.383 和 0.410。同时，晴天条件下光照强度与发电功率的相关系数为 0.9930，高于阴天条件下的 0.9866。

虽然晴天光照强度的绝对标准差更高，但这是因为晴天的整体光照水平明显更高。从相对波动角度看，阴天光照变化更不稳定，且光照强度与发电功率之间的关系更弱。这些统计差异能够解释阴天条件下 R² 较低、MAE 与 RMSE 较高的现象。

全天气数据集包含夜间和低功率记录，并拥有更多训练样本，因此其良好预测性能需要结合样本规模与运行范围进行解释。

输出文件包括：

- `results/weather_analysis.csv`
- `results/weather_comparison.png`
- `results/weather_analysis_conclusion.txt`
- `results/weather_statistics.csv`
- `results/weather_statistics.png`

运行天气影响分析：

```bash
python src/weather_impact_analysis.py
```

```bash
python scripts/weather_impact_analysis.py
```

# 模块1对模块2的支持

模块1用于预测正常天气与正常运行状态下的 Expected Normal Power Output（预期正常发电功率）。模块2可以将实际测量功率与模块1预测的正常功率进行比较：

```text
prediction residual = actual power - expected normal power
```

当 Prediction Residual（预测残差）持续较大时，可能表示系统存在异常运行状态，例如局部遮挡、组件积灰、逆变器问题、传感器故障或其他光伏系统故障。

通过建立晴天、阴天和全天气条件下的正常性能基准，模块2能够区分天气变化造成的正常功率波动与设备异常造成的异常功率偏差，从而减少仅由天气变化引起的误报。

当前限制是 Voltage（电压）和 Current（电流）尚未进入训练数据。后续加入真实电气测量数据后，可以进一步增强模块2区分天气变化与设备故障的能力。

# 项目结论

本项目完成了从数据准备、数据预处理、天气分类、多模型训练、模型评估到结果解释的完整流程。

主要结论如下：

1. Irradiance（光照强度）是当前正常发电功率预测中最重要的输入特征，三个天气数据集最佳模型中的相对重要性均超过 93%。
2. 不同天气条件适合不同模型：全天气条件下 Gradient Boosting 最佳，晴天条件下 Random Forest 最佳，阴天条件下 Lasso Regression 最佳。
3. 阴天数据具有更高的相对光照波动和更低的光照强度—发电功率相关性，因此其预测误差更高、预测难度更大。
4. 模块1建立的正常功率预测能够为模块2提供异常检测基准，预测值与实际值之间的较大持续偏差可作为潜在异常信号。

本项目的结论均基于当前 `results/model_metrics.csv`、特征重要性结果与天气统计结果，不假设未被数据支持的模型优势或特征影响。

# 后续工作

后续研究可从以下方向继续扩展：

1. 将真实 PVDAQ 样本中的 Voltage（电压）和 Current（电流）字段映射到统一训练数据。
2. 使用更多日期和更多站点的真实光伏数据替代或补充模拟数据。
3. 增加 Moderate（中等天气）、雨天、高温与低温等运行场景。
4. 使用交叉验证与超参数优化进一步验证模型稳定性。
5. 建立模块2异常检测流程，并根据预测残差设置异常阈值。
6. 分析遮挡、积灰、逆变器异常和传感器异常等不同故障类型。

课程汇报时可以概括为：

> 本项目通过多天气数据集和多模型比较，建立了正常状态下的光伏发电功率预测基准。实验结果表明，光照强度是最重要的预测特征，不同天气条件下最佳模型不同，阴天条件具有更高预测难度。模块1生成的正常功率预测结果可以进一步支持模块2的异常检测。
