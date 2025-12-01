import os
import cv2
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# ================= 配置参数 =================
IMG_HEIGHT = 256
IMG_WIDTH = 256
DEFAULT_MODEL_PATH = 'segmentation/logs/best_unet.pth' 
DEFAULT_OUTPUT_DIR = 'segmentation/signal_results'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================= 1. 核心分析功能 =================
def analyze_tumor(mask, original_shape):
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    analysis_results = []
    
    if not contours:
        return analysis_results, None

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 10: continue 
            
        x, y, w, h = cv2.boundingRect(cnt)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = x + w//2, y + h//2

        info = {
            "id": i + 1,
            "bbox": (x, y, w, h),
            "center": (cX, cY),
            "area_pixels": area
        }
        analysis_results.append(info)

    return analysis_results, contours

def save_text_report(file_path, image_name, stats, img_shape):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"=== 乳腺肿瘤检测报告 (PyTorch) ===\n")
        f.write(f"图片名称: {image_name}\n")
        f.write(f"原图尺寸: {img_shape[1]}x{img_shape[0]}\n")
        f.write(f"检测到的区域数量: {len(stats)}\n")
        f.write("-" * 30 + "\n")
        
        if not stats:
            f.write("未检测到明显的肿瘤区域。\n")
        
        for stat in stats:
            x, y, w, h = stat['bbox']
            f.write(f"区域 ID: {stat['id']}\n")
            f.write(f"  > 边界框: ({x}, {y}, {w}, {h})\n")
            f.write(f"  > 重心: {stat['center']}\n")
            f.write(f"  > 像素面积: {stat['area_pixels']:.2f}\n")
            f.write("-" * 30 + "\n")

# ================= 2. 模型加载与预处理 =================

def load_model(model_path):
    print(f"[INFO] Building UnetPlusPlus (efficientnet-b7)...")
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b7",
        encoder_weights=None, 
        in_channels=3,
        classes=1,
        activation=None
    )
    
    print(f"[INFO] Loading weights from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    state_dict = torch.load(model_path, map_location=DEVICE,weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

def get_preprocessing():
    return A.Compose([
        A.Resize(IMG_HEIGHT, IMG_WIDTH),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# ================= 3. 主预测流程 =================
def predict_and_save(image_path, model_path, output_dir):
    # 0. 准备输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filename_base = os.path.basename(image_path).split('.')[0]

    # 1. 加载模型
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    # 2. 读取图片
    if not os.path.exists(image_path):
        print(f"[ERROR] Image path '{image_path}' not found.")
        return
        
    original_img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    original_img_rgb = cv2.cvtColor(original_img_bgr, cv2.COLOR_BGR2RGB)
    original_h, original_w = original_img_rgb.shape[:2]
    
    # 3. 预处理
    transform = get_preprocessing()
    transformed = transform(image=original_img_rgb)
    input_tensor = transformed['image'].unsqueeze(0).to(DEVICE)

    # 4. 预测
    with torch.no_grad():
        output = model(input_tensor)
        output_prob = torch.sigmoid(output).squeeze().cpu().numpy()
    
    pred_mask = (output_prob > 0.5).astype(np.float32)
    pred_mask_full = cv2.resize(pred_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    # 5. 分析数据
    stats, contours = analyze_tumor(pred_mask_full, (original_h, original_w))
    
    # 6. 保存报告
    txt_save_path = os.path.join(output_dir, f"{filename_base}_report.txt")
    save_text_report(txt_save_path, os.path.basename(image_path), stats, (original_h, original_w))

    # =========================================================
    # 7. 绘制图像 (颜色修改部分)
    # =========================================================
    
    overlay_img = original_img_rgb.copy()
    
    # 定义红色 (在 RGB 模式下)
    COLOR_RED = (255, 0, 0) 
    
    if contours:
        # 画轮廓 (也改为红色，保持一致，或者你可以改回绿色 (0, 255, 0))
        
        cv2.drawContours(overlay_img, contours, -1, (0, 255, 0), 2)
        
        for stat in stats:
            x, y, w, h = stat['bbox']
            # --- 修改开始：将方框和文字颜色改为红色 ---
            # 绘制红框
            cv2.rectangle(overlay_img, (x, y), (x + w, y + h), COLOR_RED, 2)
            # 绘制 ID (红色)
            cv2.putText(overlay_img, f"ID:{stat['id']}", (x, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, 2)
            # --- 修改结束 ---

    # Resize 为 448x448 用于展示
    DISPLAY_SIZE = (448, 448)
    vis_mask = cv2.resize(pred_mask_full, DISPLAY_SIZE, interpolation=cv2.INTER_NEAREST)
    vis_overlay = cv2.resize(overlay_img, DISPLAY_SIZE)

    # Matplotlib 绘图
    plt.figure(figsize=(8, 4))

    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title("Predicted Mask", fontsize=10, pad=5)
    ax1.imshow(vis_mask, cmap='gray')
    ax1.axis('off')

    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title("Detection Analysis", fontsize=10, pad=5)
    ax2.imshow(vis_overlay)
    ax2.axis('off')

    plt.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.01, wspace=0.05)

    img_save_path = os.path.join(output_dir, f"{filename_base}_prediction.png")
    plt.savefig(img_save_path, bbox_inches='tight', pad_inches=0, dpi=150)
    print(f"[INFO] 结果图像已保存至: {img_save_path}")
    print(f"[INFO] 文本报告已保存至: {txt_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="输入图片路径")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="模型路径")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR, help="结果保存路径")
    
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"File not found: {args.image}")
        try:
            input_path = input("Please enter valid image path: ").strip().strip('"')
            if input_path:
                args.image = input_path
        except:
            pass

    predict_and_save(args.image, args.model, args.output)