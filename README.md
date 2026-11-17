# B04 目标本体表征与多源数据融合

面向多源异构电磁数据（A 类脉冲 / B 类航迹 / 文本资料）的脉冲分选、型号与状态识别、跨平台融合、A-B 跨源关联、本体化表征与态势报告生成。

- 技术方案：[docs/技术方案.md](docs/技术方案.md)
- 实验记录：[docs/实验记录.md](docs/实验记录.md)

## 流水线（src/b04）
| 模块 | 作用 |
|---|---|
| `io.py` | 按表头自动识别 A/B 类 CSV、字段兼容（`lontitude` 拼写）、结果写出与格式自检 |
| `cpi.py` | L2：单节点交错脉冲流 → CPI（RF/PW/幅度/DOA 极差门限 + 固定/抖动 PRI，容忍漏脉冲） |
| `recognize.py` | L4 型号（CPI 级 GBDT + 对象级对数似然投票）、L5 状态（驻留特征 GBDT 两级：搜索/跟踪、粗跟/精跟） |
| `tracking.py` | L3 节点内 CPI 串接、L6 跨节点同源 CPI 匹配与约束融合、对象级合并（全局发射时序无冲突） |
| `association.py` | L7 A-B 关联：方位残差命中率 + 领域/散射等级规则 + 匈牙利指派 |
| `pipeline.py` | 单场景端到端 |
| `evaluate.py` | 本地评分器（ARI / 成对 F1 / 准确率 / 关联准确率） |

`config/knowledge.yaml` 为从 D2/D3 抽取的型号—平台—领域—散射等级知识；`models/b04_models.joblib` 为训练好的模型。

## 运行
```bash
pip install -r requirements.txt
# 推理（评测入口相同）
DATA_ROOT=<测试数据目录> python src/main.py --task all --test --out-dir ./output
# 训练 + 本地评估（--split-s 300：前 300 s 训练、后段评估；默认模拟 3% 漏脉冲）
python src/train.py --data-root <训练数据目录> --split-s 300
python src/train.py --data-root <训练数据目录> --save models/b04_models.joblib
# 打包提交
python tools/pack.py --team <队名> --version v1
```

## Docker
```bash
docker build -t b04 .
docker run --rm -v <测试数据目录>:/input:ro -v $(pwd)/output:/output b04 --task all --test --out-dir /output
```

原始数据与赛事资料不纳入版本库（见 .gitignore）。
