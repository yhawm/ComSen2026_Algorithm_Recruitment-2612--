from ultralytics import YOLO
import os

def main():
    # 确保输出文件夹存在
    os.makedirs('../output/yolo', exist_ok=True)

    # ================= 修改模型读取路径 =================
    model_path = 'runs/output/yolo/drone_training/weights/best.pt'
    if not os.path.exists(model_path):
        model_path = 'F:/py/ComSen2026_Algorithm_Recruitment/task-4/runs/output/yolo/drone_training/weights/best.pt'

    model = YOLO(model_path)
    print(f"成功加载模型: {model_path}")

    print("开始评估模型，生成混淆矩阵和指标数据...")
    # 在验证集上评估，强制保存混淆矩阵，并输出到指定目录
    metrics = model.val(
        data='../drone_dataset/data.yaml',
        split='val',           # 在验证集上评估（如果要看测试集，改为 'test'）
        save_json=True,
        plots=True,            # 生成混淆矩阵图片
        project='../output/yolo', # 强制保存到 output 目录
        name='drone_eval',     # 结果保存在 output/yolo/drone_eval 下
        exist_ok=True,
        workers=0              # Windows 下必须设为 0
    )

    print("\n评估完成！混淆矩阵和评估结果已保存到 output/yolo/drone_eval 文件夹。")
    print("终端打印的 mAP50 和 mAP50-95 数据请截图保存！")

# ================= 核心保护块 =================
if __name__ == '__main__':
    main()