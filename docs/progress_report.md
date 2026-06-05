# PV Project Progress Report

这个文件用于展示当前项目每个阶段已经完成的成果，适合给老师检查进度时使用。

## 1. 数据样本准备

当前已经准备了两类数据：

### 模拟数据

- 文件：`data/simulated_pv_data.csv`
- 行数：8,760 行
- 含义：模拟一年逐小时光伏运行数据，用于保证项目在没有真实数据时也能完整运行。
- 字段：
  - `irradiance`
  - `ambient_temperature`
  - `module_temperature`
  - `humidity`
  - `wind_speed`
  - `hour`
  - `day_of_year`
  - `power_output`

### 真实数据样本

- 文件：`data/real_samples/pvdaq_system_10_2023_01_01.csv`
- 行数：1,440 行
- 数据来源：NREL / DOE PVDAQ public dataset
- 站点：PVDAQ system 10, Golden, Colorado, USA
- 时间范围：2023-01-01，一天分钟级数据
- 主要字段：
  - `measured_on`: 测量时间
  - `ac_power__423`: 交流输出功率
  - `dc_power__422`: 直流功率
  - `ambient_temp__428`: 环境温度
  - `module_temp_1__429`, `module_temp_2__430`, `module_temp_3__431`: 组件温度
  - `poa_irradiance__421`: 平面阵列辐照度

## 2. 缺失值检查

- 脚本：`src/preprocess_data.py`
- 输出：`results/missing_values.csv`
- 当前结果：模拟数据中所有核心字段缺失值数量为 0。

这个结果说明：当前数据样本可以进入后续清洗和建模阶段，不需要先做缺失值填补。

## 3. 异常值删除

- 脚本：`src/preprocess_data.py`
- 输入：`data/simulated_pv_data.csv`
- 输出：`data/clean_dataset.csv`

当前清洗结果：

- 原始数据：8,760 行
- 清洗后数据：8,382 行
- 删除行数：378 行

删除逻辑包括：

- 删除缺失值。
- 删除不符合物理意义的值，例如负辐照度、湿度超过 100%、小时不在 0 到 23 之间等。
- 使用 IQR 方法删除极端统计异常值。

## 4. 数据分布可视化

- 输出图：`results/data_distribution.png`

这张图对比了清洗前后关键字段的分布，包括：

- irradiance
- ambient_temperature
- module_temperature
- humidity
- wind_speed
- power_output

展示时可以说明：这一步用于检查数据是否存在明显异常，以及清洗后数据是否仍然保留合理分布。

## 5. 清洗后数据集

- 文件：`data/clean_dataset.csv`
- 用途：后续 AI 模型训练的主要输入数据。

训练脚本 `src/train_model.py` 已经设置为使用 `clean_dataset.csv` 进行模型训练。

## 6. 多模型初步训练与比较

- 脚本：`src/train_model.py`
- 当前模型：
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
  - `results/model_comparison.png`
  - `results/prediction_results.csv`
  - `results/predicted_vs_actual.png`

当前阶段模型属于初步训练和对比，用于观察不同算法在光伏功率预测任务上的表现。后续还可以继续调参、换真实数据、增加更多天气特征。

## 7. 当前模型比较结果

当前结果显示，`Gradient Boosting` 在测试集上表现最好：

- MAE: 0.7014
- RMSE: 1.1550
- R2: 0.9973

模型对比图可以查看：

```text
results/model_comparison.png
```

## 8. 汇报话术

可以这样讲：

> 我目前已经完成了光伏数据样本准备和数据预处理。项目中先用模拟数据保证完整流程可以运行，同时我也找到了 NREL / DOE PVDAQ 的真实光伏数据样本。预处理阶段已经完成缺失值检查、异常值删除和数据分布可视化，并输出了 clean_dataset.csv。现在模型训练已经从 baseline 扩展到多模型比较，包括线性模型、树模型、集成模型、KNN 和 SVR。当前 Gradient Boosting 表现最好，下一步会把真实数据进一步整理成统一字段，并继续优化模型效果。

## 9. 下一步计划

1. 把 PVDAQ 真实数据字段映射到项目统一字段。
2. 寻找或补充真实数据中的湿度、风速字段。
3. 使用真实数据训练初版模型。
4. 对比模拟数据模型和真实数据模型的效果。
5. 分析预测值与实际值差异，用于后续异常检测。
