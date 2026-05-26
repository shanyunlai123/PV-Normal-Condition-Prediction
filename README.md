# PV Normal Condition Power Prediction

这是一个用于预测正常状态下光伏发电功率的 AI 项目。目前项目进度适合下周五向老师展示：已经准备了光伏数据样本，完成了数据预处理流程，并开始进行基础 AI 模型训练。

## 1. 项目结构

```text
PV/
  data/
    simulated_pv_data.csv
    clean_dataset.csv
  src/
    generate_data.py
    preprocess_data.py
    train_model.py
    predict.py
  models/
    best_pv_power_model.joblib
  results/
    missing_values.csv
    data_distribution.png
    prediction_results.csv
    predicted_vs_actual.png
    model_metrics.csv
  notebooks/
  README.md
  requirements.txt
```

## 2. 安装依赖

```bash
pip install -r requirements.txt
```

## 3. 当前阶段目标

下周五展示的重点不是最终完成所有模型优化，而是展示项目正在按正确流程推进：

1. 已经准备光伏发电数据样本。
2. 已经完成数据字段整理和数据质量检查。
3. 已经完成缺失值检查、异常值删除和数据分布可视化。
4. 已经输出可用于模型训练的 `clean_dataset.csv`。
5. AI 模型已经开始初步训练，当前包含 Linear Regression 和 Random Forest 两个 baseline 模型。

## 4. 直接运行

生成模拟数据：

```bash
python src/generate_data.py
```

进行数据预处理：

```bash
python src/preprocess_data.py
```

训练 Linear Regression 和 Random Forest，并生成结果文件：

```bash
python src/train_model.py
```

运行单条样本预测：

```bash
python src/predict.py
```

注意：如果存在 `data/clean_dataset.csv`，`train_model.py` 会优先使用清洗后的数据进行训练。

## 5. 数据是什么

数据集是模拟的一年逐小时光伏运行数据。它表示在正常运行状态下，光伏电站受到太阳辐照度、温度、湿度、风速和时间变化影响后的发电功率。

字段包括：

- `irradiance`: 太阳辐照度，单位约为 W/m2。通常越高，发电功率越高。
- `ambient_temperature`: 环境温度，单位为摄氏度。
- `module_temperature`: 光伏组件温度，单位为摄氏度。组件过热会降低发电效率。
- `humidity`: 空气湿度，单位为百分比。
- `wind_speed`: 风速，单位约为 m/s。风可以降低组件温度。
- `hour`: 一天中的小时，用于表达日出、正午、日落等日内规律。
- `day_of_year`: 一年中的第几天，用于表达季节变化。
- `power_output`: 光伏发电功率，单位为 kW，是模型要预测的目标值。

## 6. 数据预处理做了什么

预处理脚本是 `src/preprocess_data.py`，它完成以下工作：

1. 读取 `data/simulated_pv_data.csv`。
2. 检查每个字段的缺失值，并输出 `results/missing_values.csv`。
3. 删除缺失值。
4. 删除不符合物理意义的异常值，例如负辐照度、湿度超过 100%、小时不在 0 到 23 之间等。
5. 使用 IQR 方法删除极端统计异常值。
6. 输出清洗后的数据 `data/clean_dataset.csv`。
7. 生成数据分布图 `results/data_distribution.png`，用于展示清洗前后的数据分布。

## 7. AI 模型怎么训练

训练脚本会把数据分成训练集和测试集：

- 训练集：用于让模型学习天气、时间和发电功率之间的关系。
- 测试集：模型没有见过的数据，用来检验预测能力。

本项目训练两个模型：

- `Linear Regression`: 线性回归，简单、容易解释，适合作为基准模型。
- `Random Forest`: 随机森林，可以学习更复杂的非线性关系，通常比线性模型更适合光伏功率预测。

输入特征是：

```text
irradiance, ambient_temperature, module_temperature, humidity, wind_speed, hour, day_of_year
```

预测目标是：

```text
power_output
```

## 8. 模型结果怎么看

训练完成后会生成三个主要结果文件：

- `results/model_metrics.csv`: 两个模型的评估指标。
- `results/prediction_results.csv`: 每条测试样本的真实功率和预测功率。
- `results/predicted_vs_actual.png`: 真实值与预测值的散点图。

评估指标含义：

- `MAE`: 平均绝对误差。表示预测值平均偏离真实值多少 kW，越小越好。
- `RMSE`: 均方根误差。对大误差更敏感，越小越好。
- `R2`: 决定系数。越接近 1，说明模型解释能力越强。

看图时可以这样理解：

- 横轴是真实发电功率。
- 纵轴是模型预测发电功率。
- 黑色对角线代表完美预测。
- 点越靠近黑色对角线，说明预测越准确。

## 9. 下周五进度展示时应该怎么讲

可以按这个顺序讲：

1. 研究目标：我想建立一个 AI 模型，用天气和运行条件预测正常状态下的光伏发电功率。
2. 当前进度：目前已经完成数据样本准备和数据预处理，模型训练已经开始做 baseline。
3. 数据字段：输入包括辐照度、环境温度、组件温度、湿度、风速、小时和年内天数，输出是发电功率。
4. 预处理工作：我检查了缺失值，删除了异常值，并生成了清洗后的 `clean_dataset.csv`。
5. 可视化展示：展示 `data_distribution.png`，说明我已经检查了数据分布。
6. 初步模型：目前先用 Linear Regression 和 Random Forest 做初步训练，后面会继续调参、换真实数据或加入更多特征。
7. 下一步计划：继续优化模型，比较不同算法效果，并研究预测值和实际值偏差是否可以用于发现异常发电情况。

一句话总结：

> 当前阶段我已经完成了数据准备和预处理，正在进入 AI 模型初步训练阶段，后续会继续优化预测精度和结果解释。
