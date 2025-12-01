import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import os
import torch.nn.functional as F
import argparse
import types  # <--- [新增] 用于动态替换方法

# 设置 Matplotlib 后端为 Agg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -------------------------------------------------
# 1. 配置参数
# -------------------------------------------------
BASE_PATH = ""
# 确保这里指向正确的权重路径
MODEL_PATH = os.path.join(BASE_PATH, "classification/logs/DenseNet121_best.pth")
DEFAULT_IMAGE_PATH = os.path.join(BASE_PATH, "classification/workings/test/malignant/malignant (39).png")
OUTPUT_DIR = os.path.join(BASE_PATH, "classification/single_results")
CLASS_NAMES = ['benign', 'malignant', 'normal']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# 2. Grad-CAM 工具类 (保持不变)
# -------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach()
        
        for i in range(activations.shape[1]):
            activations[0, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        
        if torch.max(heatmap) == 0:
            heatmap = heatmap
        else:
            heatmap /= torch.max(heatmap)
        
        return heatmap.cpu().numpy(), output

# -------------------------------------------------
# 3. 辅助函数 (关键修改在 get_model)
# -------------------------------------------------
def get_model():
    print(f"[INFO] Loading DenseNet121 from {MODEL_PATH}...")
    
    # 1. 初始化模型
    model = models.densenet121(weights=None)
    
    # 2. 重构分类器
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, len(CLASS_NAMES))
    )
    
    # 3. 加载权重 (修复 weights_only 警告)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
    # 显式设置 weights_only=False 以兼容旧版保存习惯，或者 True 如果只有 state_dict
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    
    # --- [关键修复] 开始 ---
    # 目的：覆盖 DenseNet 原生的 forward 方法，将 F.relu(..., inplace=True) 改为 False
    # 否则 Hook 会因为变量被就地修改而报错
    
    def patched_forward(self, x):
        features = self.features(x)
        # 这里改为 inplace=False
        out = F.relu(features, inplace=False) 
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

    # 将新方法绑定到模型实例上
    model.forward = types.MethodType(patched_forward, model)
    # --- [关键修复] 结束 ---

    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    raw_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(raw_image).unsqueeze(0)
    return raw_image, input_tensor

def show_cam_on_image(img, mask):
    heatmap = cv2.resize(mask, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img

def save_text_report(save_path, img_name, pred_class, probabilities):
    with open(save_path, "w", encoding='utf-8') as f:
        f.write("=== Breast Ultrasound Classification Report ===\n")
        f.write(f"Image Name: {img_name}\n")
        f.write(f"Prediction: {pred_class}\n")
        f.write("-" * 30 + "\n")
        f.write("Probabilities:\n")
        for cls, prob in zip(CLASS_NAMES, probabilities):
            f.write(f"  {cls:<10}: {prob*100:.2f}%\n")
        f.write("-" * 30 + "\n")
    print(f"[INFO] Report saved: {save_path}")

# -------------------------------------------------
# 4. 主流程
# -------------------------------------------------
def main(image_path):
    if not os.path.exists(image_path):
        print(f"[ERROR] Image file does not exist: {image_path}")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        # 1. 加载模型 (已修复 inplace 问题)
        model = get_model()
        
        # 2. 设置 Hook 目标层
        target_layer = model.features.norm5 
        grad_cam = GradCAM(model, target_layer)

        # 3. 推理与 CAM
        raw_image, input_tensor = preprocess_image(image_path)
        input_tensor = input_tensor.to(device)

        print(f"[INFO] Analyzing image: {os.path.basename(image_path)}")
        heatmap, output = grad_cam(input_tensor)

        # 4. 预测结果
        probs = F.softmax(output, dim=1).cpu().detach().numpy()[0]
        pred_idx = np.argmax(probs)
        pred_label = CLASS_NAMES[pred_idx]
        
        print(f"\n>>> Prediction: {pred_label} ({probs[pred_idx]*100:.2f}%)")

        # 5. 可视化
        DISPLAY_SIZE = (448, 448)
        raw_img_np = np.array(raw_image)
        vis_img = show_cam_on_image(raw_img_np, heatmap)
        
        vis_raw = cv2.resize(raw_img_np, DISPLAY_SIZE)
        vis_cam = cv2.resize(vis_img, DISPLAY_SIZE)

        # 6. 保存图片
        filename = os.path.basename(image_path).split('.')[0]
        plt.figure(figsize=(8, 4)) 
        
        ax1 = plt.subplot(1, 2, 1)
        ax1.set_title("Original Image", fontsize=10, pad=5)
        ax1.imshow(vis_raw)
        ax1.axis('off')
        
        ax2 = plt.subplot(1, 2, 2)
        ax2.set_title(f"Grad-CAM: {pred_label}", fontsize=10, pad=5)
        ax2.imshow(vis_cam)
        ax2.axis('off')
        
        plt.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.01, wspace=0.05)
        
        img_save_path = os.path.join(OUTPUT_DIR, f"{filename}_result.png")
        plt.savefig(img_save_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()
        print(f"[INFO] Result image saved: {img_save_path}")
        
        # 7. 保存报告
        txt_save_path = os.path.join(OUTPUT_DIR, f"{filename}_report.txt")
        save_text_report(txt_save_path, os.path.basename(image_path), pred_label, probs)

    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE_PATH)
    args = parser.parse_args()
    main(args.image)