电表数字自动读取项目 · 使用指南

项目根目录： meter_reading/

运行环境： macOS / Linux / Windows (推荐 Conda 或 venv)

主要语言： Python 3.11


1. 项目简介

本仓库实现了一个 端到端的电表数字识别系统，包括：

阶段	目标	脚本
预处理	去光照、校正畸变	preprocess.py
目标检测训练	学习框出数字区域	train_detector.py + csv2yolo.py
检测推理	ROI 自动裁剪	detect_region.py
切分数字	ROI → 单字符	split_digits.py
分类器训练	0-9 识别	train_classifier.py
最终读数	拼接字符串	classify_digits.py
评估	统计准确率	evaluate.py

📌 快捷跑通：bash quick_run.sh  (已写好示例脚本，见根目录)


2. 目录结构

meter_reading/
├── README.md                ← 你正在看的文档
├── requirements.txt         ← 依赖列表
├── quick_run.sh             ← 一键跑通示例（可选）
│
├── *.py                     ← 各阶段脚本
│
├── models/                  ← 训练生成的权重
│   ├── classifier_best.pth
│   └── detector/exp*/weights/best.pt
│
└── data/
    ├── raw_images/          ← 老师/现场采集原图 (.jpg)
    ├── preprocessed/        ← 预处理输出
    ├── labels.csv           ← 老师标注文件 (含 xmin/ymin/xmax/ymax)
    ├── meter_dataset/       ← 检测器训练集 (脚本自动生成)
    ├── regions/             ← 检测裁剪 ROI
    └── digits/              ← 单字符切分 & train/val 子集



3. 安装依赖

# 1. 推荐创建虚拟环境
python -m venv .venv && source .venv/bin/activate  # Linux/Mac
# Windows: python -m venv .venv && .\.venv\Scripts\activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. (可选) Apple M-series GPU 训练
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/mps

主要依赖：ultralytics (≥8.3)、opencv-python (4.10)、torch (≥2.2)、torchvision、numpy、pandas、tqdm、scikit-image、python-Levenshtein。


4. 数据准备
	1.	把老师给的原图放入 data/raw_images/。
	2.	确保 data/labels.csv 包含列：filename,xmin,ymin,xmax,ymax,number。
	3.	其余目录脚本会自动创建。


5. 完整流程

# Step 0  依赖已装好

# 1. 预处理
python preprocess.py

# 2. 训练检测器
python train_detector.py --epochs 100 --imgsz 640

# 3. 检测 + 裁剪 ROI
python detect_region.py

# 用 YOLO 训练 split_digits 模型
yolo detect train data=./data/splits/data.yaml model=yolov8n.pt imagsz=640 epoch=100

# 4. ROI → 单数字
python split_digits.py

# 5. （手动把一部分 digits 划分 train/val/0~9 后）训练分类器
python train_classifier.py

# 6. 单张推理 & 可视化
python infer.py \
	--crop_dir ./out_digits \
	--model ./models/resnet18_digits.pth \
	--out readingsV4.json

# 7. 批量评估
python evaluate.py

建议：先用 10–20 张图跑通流程，确认切分 & 分类没问题，再全量训练。



6. 联系与贡献
	•	作者：赵泽文团队
	•	问题反馈：Issue / PR / QQ 群 / WeChat 均可
	•	欢迎对脚本或流程提出改进建议！


🌟 Tip：若团队成员只想“看结果”，直接执行 quick_run.sh 即可自动跑完整 pipeline（需提前准备 raw_images 和 labels.csv）。


© 2025 BUPT Computer Science Project