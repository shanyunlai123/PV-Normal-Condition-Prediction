# PV Power Prediction

这是一个用于预测正常状态下光伏发电功率的 Python 项目。项目使用光伏运行数据和天气数据训练 AI 模型，预测 `normal PV power generation performance`。

如果没有真实数据，可以先运行模拟数据脚本生成可训练的数据集。整个项目可以从数据生成、模型训练、评估到单条预测完整运行。

## 项目结构

```text
PV/
  data/                 # 原始或模拟数据
  src/                  # 源代码
  models/               # 训练后的模型文件
  results/              # 评估指标和图表
  notebooks/            # 实验笔记本目录
  README.md
  requirements.txt
```

## 安装依赖

```bash
pip install -r requirements.txt
```

如果 `xgboost` 安装失败，训练脚本会自动退回到 scikit-learn 的随机森林模型。

## 快速开始

1. 生成模拟光伏数据：

```bash
python src/generate_data.py
```

2. 训练并评估模型：

```bash
python src/train_model.py
```

3. 使用训练好的模型做一次示例预测：

```bash
python src/predict.py
```

## 数据字段

模拟数据文件位于 `data/simulated_pv_data.csv`，主要字段包括：

- `timestamp`: 时间戳
- `irradiance_w_m2`: 太阳辐照度
- `ambient_temp_c`: 环境温度
- `module_temp_c`: 组件温度
- `wind_speed_m_s`: 风速
- `humidity_pct`: 湿度
- `cloud_cover_pct`: 云量
- `hour`: 小时
- `day_of_year`: 年内第几天
- `power_kw`: 光伏发电功率目标值

## 模型输出

运行训练脚本后会生成：

- `models/pv_power_model.joblib`: 模型和特征列
- `results/metrics.json`: MAE、RMSE、R2 等指标
- `results/prediction_vs_actual.png`: 预测值与真实值对比图
- `results/feature_importance.png`: 特征重要性图

## 使用真实数据

如果你有真实光伏运行数据，请整理为 CSV，并尽量包含天气、时间和目标功率字段。可以修改 `src/train_model.py` 中的 `DATA_PATH` 和 `FEATURE_COLUMNS`，让训练脚本读取真实数据。
