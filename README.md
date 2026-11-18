# B04 目标本体表征与多源数据融合

面向多源异构电磁数据（A 类脉冲 / B 类航迹 / 文本资料）的脉冲分选、型号与状态识别、跨平台融合、A-B 跨源关联、本体化表征与态势报告生成。

- 技术方案：[docs/技术方案.md](docs/技术方案.md)（内部路线）；提交用技术方案报告正文 [docs/技术方案报告/正文.md](docs/技术方案报告/正文.md)，`python docs/技术方案报告/make_figures.py` 生成插图，`python tools/build_tech_report.py --team 同频队 [--leader 姓名 --unit 单位 --date 2026年12月]` 套官方模板输出正式版/盲审版 docx+PDF 到 dist/技术方案/（需本机 Word）
- 实验记录：[docs/实验记录.md](docs/实验记录.md)

## 流水线（src/b04）
| 模块 | 作用 |
|---|---|
| `io.py` | 按表头自动识别 A/B 类 CSV、字段兼容（`lontitude` 拼写）、结果写出与格式自检 |
| `cpi.py` | L2：单节点交错脉冲流 → CPI（RF/PW/幅度/DOA 极差门限 + 固定/抖动 PRI，容忍漏脉冲） |
| `fingerprint.py` | L4 型号：生成式波形指纹似然（RF/PW/PRI 核密度 + 抖动，无场景先验），对象级证据累积 |
| `anchor.py` | 驻留组（跨节点同源 + CPI 首尾相接）、B 航迹几何锚定、时序冲突消解、标签传播 |
| `recognize.py` | L5 状态：两级 GBDT（搜索/跟踪 → 粗跟/精跟），首尾相接/多节点/固定重访/跟踪段序号特征，按发射平均 |
| `tracking.py` | L3 节点内 CPI 串接、L6 跨节点同源 CPI 匹配与约束融合、对象级合并（型号似然 + 全局发射时序无冲突）、小对象并入 |
| `association.py` | L7 A-B 关联：方位残差命中率 + 领域/散射等级规则 + 匈牙利指派 |
| `situation.py` | 态势要素抽取：运动方式、伴随/接近、跟踪照射、观测关系、把握度（事实 JSON） |
| `ontology.py` | 电磁目标本体（RDF/OWL）→ Turtle / JSON-LD / 交互式图谱 HTML |
| `report.py` / `agent.py` | D4 结构态势报告（HTML→PDF）；可选大模型研判（环境变量注入、事实锁定校验、失败降级） |
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
# 态势要素 + 本体图谱 + 态势报告（输出到 ./output/report/）
DATA_ROOT=<测试数据目录> python src/main.py --task report --out-dir ./output
powershell -File tools/html2pdf.ps1 -Html output/report/<场景>_<版本>_situation_report.html
# 打包提交：代码包 + A 榜结果包（结果包先按提交规则自检，不通过则拒绝打包）
python tools/pack.py --team 同频队 --version v8
python tools/pack_results.py --team 同频队 --version v8 --data-root <测试数据目录>
# Linux/Docker 验证（在 WSL Ubuntu 中执行，流程同评测服务）
bash tools/validate_docker.sh dist/同频队_code_v8.zip <测试数据目录>
```

## Docker
```bash
docker build -t b04 .
docker run --rm -v <测试数据目录>:/input:ro -v $(pwd)/output:/output b04 --task all --test --out-dir /output
```

原始数据与赛事资料不纳入版本库（见 .gitignore）。
