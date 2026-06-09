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

## 每一步展示什么

详细展示材料在：

```text
docs/progress_report.md
```


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
16. 特征重要性结果：`results/feature_importance.csv`
17. 特征重要性图：`results/feature_importance.png`
18. 天气影响比较表：`results/weather_analysis.csv`
19. 天气影响比较图：`results/weather_comparison.png`
20. 自动天气分析结论：`results/weather_analysis_conclusion.txt`

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

## Feature Importance Analysis

运行 `src/feature_importance_analysis.py` 可以使用已经训练完成的最佳模型分析变量对 PV power prediction 的影响程度。

分析规则：

- 如果最佳模型是 Random Forest、Gradient Boosting、Extra Trees 或 Decision Tree，则使用模型的 `feature_importances_`。
- 如果最佳模型是 Linear Regression、Ridge Regression 或 Lasso Regression，则使用模型系数的绝对值。
- 所有结果会自动按重要性从高到低排序。

当前最佳模型是全量天气数据集上的 `Gradient Boosting`。当前结果显示：

```text
irradiance:  99.87%
temperature:  0.13%
voltage:      not used in current model
current:      not used in current model
```

因此，当前模型中对 PV power prediction 影响最大的变量是 `irradiance`。

`temperature` 由 `ambient_temperature` 和 `module_temperature` 的重要性合并得到。当前训练数据没有将 voltage 和 current 作为模型输入，因此分析结果会明确标记它们为 `used_in_model=False`，不会错误地将其解释为低影响变量。

**English interpretation:** Irradiance is the most important feature in the current best model. This is expected because PV modules convert incoming solar radiation into electrical power, so changes in irradiance directly change the available power output. The result is consistent with the physical behavior of a PV system. Temperature has a smaller secondary effect because higher module temperature can reduce conversion efficiency. Voltage and current are reported as not used because they are not input features in the current trained model.

输出文件：

- `results/feature_importance.csv`
- `results/feature_importance.png`

## Weather Impact Analysis

运行 `src/weather_impact_analysis.py` 可以读取 all_weather、sunny 和 cloudy 三个数据集的模型结果，并比较每个数据集最佳模型的 R²、MAE 和 RMSE。

分析流程：

- 从 `results/model_metrics.csv` 读取三个数据集的全部模型结果。
- 自动选择每个数据集 RMSE 最低的最佳模型。
- 按预测准确度自动排序。
- 生成 R² 和 MAE comparison chart。
- 自动判断哪种天气预测最准确、哪种天气预测最困难，以及天气变化对模型性能的影响。

当前结果：

```text
all_weather: R2 0.9973, MAE 0.7014, RMSE 1.1550
sunny:       R2 0.9927, MAE 1.1837, RMSE 1.5607
cloudy:      R2 0.9738, MAE 1.3266, RMSE 1.6843
```

分析结论：

- 当前 `all_weather` 数据集的整体预测最准确，最佳模型是 Gradient Boosting。
- `cloudy` 数据集预测最困难，最佳模型是 Lasso Regression。
- 阴天条件下辐照度和功率变化更不稳定，模型误差更高，说明天气变化会明显影响 PV power prediction 的性能。
- all_weather 数据集包含更多训练样本，因此其结果也受到样本数量优势影响；天气影响分析应同时结合数据集大小理解。

**English interpretation:** The all-weather dataset gives the best overall prediction result, while the cloudy dataset is the most difficult. Sunny conditions are more regular because irradiance and power usually follow a smoother daily pattern. Cloudy conditions introduce rapid and irregular irradiance changes, making the relationship between weather inputs and power output harder to predict. The larger all-weather dataset also benefits from having more training samples, so dataset size should be considered when interpreting the comparison.

输出文件：

- `results/weather_analysis.csv`
- `results/weather_comparison.png`
- `results/weather_analysis_conclusion.txt`

## Discussion

The analysis shows that PV power prediction depends strongly on both the selected input features and the operating weather condition. Irradiance dominates the feature importance results because it represents the solar energy available to the PV modules. Temperature has a smaller but physically meaningful influence through module efficiency.

Model performance also changes across weather conditions. Random Forest performs best for sunny data, while Lasso Regression performs best for cloudy data. This suggests that no single model structure is guaranteed to be optimal for every weather scenario. Cloudy conditions remain more difficult because short-term irradiance fluctuations create a less stable power-generation pattern.

Voltage and current are available in the real PVDAQ sample but are not yet included in the current training dataset. A future extension can map these real measurements into the model input features and repeat the feature importance analysis.

## Key Findings

- **Most important feature:** Irradiance is the most important feature for PV power prediction, contributing approximately 99.87% of the normalized importance among the requested features currently used by the model.
- **Best model by weather condition:** Gradient Boosting performs best on `all_weather`, Random Forest performs best on `sunny`, and Lasso Regression performs best on `cloudy`.
- **Weather and prediction accuracy:** Weather conditions affect prediction accuracy because cloudy conditions create faster and less predictable irradiance changes than sunny conditions.
- **Support for Module 2 anomaly detection:** Module 1 establishes the expected normal power output under different weather conditions. In Module 2, a large difference between expected power and measured power can be used as an anomaly signal for faults, shading, soiling, or abnormal system operation.

## Feature Importance Analysis Evidence

The evidence script reads the existing `results/model_metrics.csv`, selects the best existing model for each dataset, and applies the appropriate importance method:

- Tree models use `feature_importances_`.
- Linear models use absolute coefficients from the existing standardized pipeline.

| Dataset | Best model | Irradiance importance | Temperature importance | Method |
|---|---|---:|---:|---|
| all_weather | Gradient Boosting | 99.88% | 0.12% | `feature_importances_` |
| sunny | Random Forest | 99.39% | 0.61% | `feature_importances_` |
| cloudy | Lasso Regression | 93.96% | 6.04% | standardized absolute coefficients |

Irradiance is the most important used feature for all three datasets. This is physically reasonable because PV power is primarily determined by the solar energy reaching the modules. Temperature has a secondary influence because module efficiency changes with temperature.

The cloudy model assigns a larger relative influence to temperature than the sunny and all-weather models. This supports the model results: cloudy power prediction contains more secondary variation, while sunny and all-weather predictions are dominated more strongly by irradiance.

Voltage and current are not input features in the current training datasets, so their importance cannot yet be measured. They are marked as `used_in_model=False` in the output files rather than being assigned an unsupported interpretation.

Evidence files:

- `results/feature_importance_all_weather.csv`
- `results/feature_importance_sunny.csv`
- `results/feature_importance_cloudy.csv`
- `results/feature_importance_comparison.png`

## Weather Impact Evidence

The following evidence is calculated directly from the existing all-weather, sunny, and cloudy datasets:

| Dataset | Irradiance mean | Irradiance std | Irradiance CV | Power mean | Power std | Irradiance-power correlation |
|---|---:|---:|---:|---:|---:|---:|
| all_weather | 194.169 | 244.163 | 1.257 | 18.038 | 22.573 | 0.9973 |
| sunny | 560.679 | 214.581 | 0.383 | 51.514 | 18.737 | 0.9930 |
| cloudy | 275.080 | 112.726 | 0.410 | 26.340 | 10.621 | 0.9866 |

Sunny data has a lower relative irradiance variation than cloudy data (`CV 0.383` versus `0.410`) and a stronger irradiance-power correlation (`0.9930` versus `0.9866`). Although sunny irradiance has a larger absolute standard deviation because its irradiance level is much higher, cloudy irradiance varies more relative to its mean. This weaker and relatively more variable relationship helps explain why cloudy prediction has lower R² and higher errors.

The all-weather dataset includes night and low-power records and has substantially more rows. Its strong result should therefore be interpreted together with its larger sample size and broad operating range.

Evidence files:

- `results/weather_statistics.csv`
- `results/weather_statistics.png`

## Model Result Discussion

The following statements use the existing `results/model_metrics.csv` values and do not introduce new models:

- **All Weather:** Gradient Boosting performs best with `MAE 0.7014`, `RMSE 1.1550`, and `R² 0.9973`. Gradient Boosting can combine many small nonlinear corrections across the broad all-weather operating range, which may explain its advantage on this larger and more varied dataset.
- **Sunny:** Random Forest performs best with `MAE 1.1837`, `RMSE 1.5607`, and `R² 0.9927`. Sunny data has a strong irradiance-power relationship, while Random Forest can still capture smaller nonlinear effects from temperature and time features.
- **Cloudy:** Lasso Regression performs best with `MAE 1.3266`, `RMSE 1.6843`, and `R² 0.9738`. Cloudy data has the weakest irradiance-power correlation and the highest relative irradiance variability of the two daytime weather groups. Lasso regularization may reduce overfitting on this smaller, less stable dataset.

Different weather conditions can prefer different models because they create different relationships between irradiance, temperature, and power. The results show that model selection should be supported by measured errors for each operating condition rather than assuming one model is always best.

## Connection to Anomaly Detection

Module 1 predicts the expected normal PV power output from weather and operating inputs. Module 2 can compare that expected output with the actual measured output:

```text
prediction residual = actual power - expected normal power
```

A large or persistent residual may indicate abnormal operating conditions such as shading, soiling, inverter problems, sensor faults, or other PV system faults. Weather-specific evidence improves this process because Module 2 can use an appropriate normal-performance baseline for sunny and cloudy conditions, reducing false anomaly alarms caused only by weather changes.

The remaining limitation is that voltage and current are not yet included in the current training datasets. Adding those real measurements later would provide stronger electrical evidence for distinguishing weather-driven changes from equipment faults.

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

运行最佳模型特征重要性分析：

```bash
python src/feature_importance_analysis.py
```

运行天气影响分析：

```bash
python src/weather_impact_analysis.py
```

运行逐数据集 Feature Importance 证据分析：

```bash
python scripts/feature_importance_analysis.py
```

运行天气统计证据分析：

```bash
python scripts/weather_impact_analysis.py
```


> 我目前已经完成了光伏数据样本准备和数据预处理，并且不只是使用一个简单数据集训练模型。我把清洗后的数据进一步划分成全量天气、晴天和阴天数据集，然后在每个数据集上分别训练 Linear Regression、Random Forest、Gradient Boosting、SVR 等多个模型。初步结果显示，不同天气条件下最佳模型不同，这说明后续可以针对不同天气场景建立更细分的预测模型。
