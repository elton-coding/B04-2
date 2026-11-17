# B04 目标本体表征与多源数据融合

面向多源异构电磁数据（A 类脉冲 / B 类航迹 / 文本资料）的脉冲分选、型号与状态识别、跨平台融合、A-B 跨源关联、本体化表征与态势报告生成。

- 技术方案：[docs/技术方案.md](docs/技术方案.md)

## 目录
```
Dockerfile / run.sh / requirements.txt   # 评测契约
src/main.py                              # 入口：--task --test --out-dir
config/                                  # 配置
output/                                  # 结果输出
docs/                                    # 方案与实验记录
```

## 本地运行
```bash
pip install -r requirements.txt
DATA_ROOT=<测试数据目录> python src/main.py --task all --test --out-dir ./output
```

## Docker
```bash
docker build -t b04 .
docker run --rm -v <测试数据目录>:/input:ro -v $(pwd)/output:/output b04 --task all --test --out-dir /output
```

原始数据与赛事资料不纳入版本库（见 .gitignore）。
