# PV Normal Condition Power Prediction

这是一个最小可展示的 AI 项目，用于演示如何根据天气和时间信息预测正常状态下的光伏发电功率。

项目会自动生成模拟光伏数据，训练两个机器学习模型，并输出预测结果、评估指标和可视化图表。

## 1. 项目结构

```text
PV/
  data/
    simulated_pv_data.csv
  src/
    generate_data.py
    train_model.py
    predict.py
  models/
    best_pv_power_model.joblib
  results/
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

## 3. 直接运行

生成模拟数据：

```bash
python src/generate_data.py
```

训练 Linear Regression 和 Random Forest，并生成结果文件：

```bash
python src/train_model.py
```

运行单条样本预测：

```bash
python src/predict.py
```

注意：如果还没有 `data/simulated_pv_data.csv`，`train_model.py` 会自动生成一份模拟数据。

## 4. 数据是什么

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

## 5. AI 模型怎么训练

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

## 6. 模型结果怎么看

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

## 7. 下周展示时应该怎么讲

可以按这个顺序讲：

1. 业务问题：光伏发电会受到太阳辐照度、温度、湿度、风速和时间影响，我们希望用 AI 预测正常状态下应该发多少电。
2. 数据输入：模型使用天气数据和时间特征，比如 `irradiance`、`module_temperature`、`hour`、`day_of_year`。
3. 预测目标：模型预测 `power_output`，也就是正常情况下的光伏发电功率。
4. 模型方法：先用 Linear Regression 做可解释的基准，再用 Random Forest 捕捉更复杂的非线性关系。
5. 评估方式：用 MAE、RMSE 和 R2 判断模型好不好。
6. 图表解读：展示 `predicted_vs_actual.png`，说明散点越接近对角线，预测越准确。
7. 实际价值：如果未来真实发电功率明显低于模型预测的正常功率，可能说明有遮挡、设备故障、积灰或其他异常，需要进一步检查。

一句话总结：

> 这个项目演示了如何用天气和运行条件训练 AI 模型，预测正常状态下光伏电站应该达到的发电功率，并用预测值和真实值的差异辅助发现潜在异常。
