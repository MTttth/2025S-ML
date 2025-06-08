"""
训练一个目标检测模型（如 YOLOv5），让它学会在一张预处理后的电表整图里，用一个矩形框自动“框”出数字读数区。
输入：标注好的训练集/验证集：data/meter_dataset/images/{train,val}， data/meter_dataset/labels/{train,val}
输出：PyTorch 模型文件（.pt）
"""