import os
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast, GradScaler
import segmentation_models_pytorch as smp
import pandas as pd
import matplotlib
# 设置后端，防止在无显示器的服务器上报错
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# ================= 配置参数 =================
DATA_PATH = 'MuRAG-CAD/Dataset_BUSI_with_GT'  # 请修改为你实际的数据集路径
LOG_DIR = 'MuRAG-CAD/segmentation/logs'                    # 日志保存目录

IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3
BATCH_SIZE = 16  # EfficientNet-B7 显存占用大，建议调小BatchSize，如果显存够大可改回 64
EPOCHS = 50
LEARNING_RATE = 1e-4
SEED = 42

# 创建日志目录
os.makedirs(LOG_DIR, exist_ok=True)
WEIGHT_SAVE_PATH = os.path.join(LOG_DIR, 'best_unet.pth')
CSV_SAVE_PATH = os.path.join(LOG_DIR, 'training_log.csv')
PLOT_SAVE_PATH = os.path.join(LOG_DIR, 'training_curves.png')

# 设置随机种子
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

set_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================= 1. 数据增强策略 =================
train_transform = A.Compose([
    A.Resize(IMG_HEIGHT, IMG_WIDTH),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(IMG_HEIGHT, IMG_WIDTH),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# ================= 2. 数据加载器 =================
def load_data_paths(root_path):
    classes = sorted(os.listdir(root_path))
    images = []
    masks = []
    
    for cls in classes:
        cls_path = os.path.join(root_path, cls)
        if not os.path.isdir(cls_path): continue
            
        files = glob.glob(os.path.join(cls_path, '*'))
        
        mask_files = [f for f in files if 'mask' in f]
        img_files = [f for f in files if 'mask' not in f]
        
        img_files.sort()
        mask_files.sort()
        
        for img_path in img_files:
            basename = os.path.basename(img_path).split('.')[0]
            matching_masks = [m for m in mask_files if basename in m]
            
            if matching_masks:
                images.append(img_path)
                masks.append(matching_masks[0]) 

    return images, masks

class BUSIDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            mask_path = self.mask_paths[idx]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)

            return image, mask.float()
        except Exception as e:
            print(f"Error loading {self.image_paths[idx]}: {e}")
            return torch.zeros((3, IMG_HEIGHT, IMG_WIDTH)), torch.zeros((1, IMG_HEIGHT, IMG_WIDTH))

# ================= 3. 损失函数与评价指标 (含 Acc) =================
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce = self.bce(pred, target)
        pred_sigmoid = torch.sigmoid(pred)
        smooth = 1e-5
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        dice_loss = 1 - dice
        return self.bce_weight * bce + (1 - self.bce_weight) * dice_loss

def calculate_metrics(pred_logits, target):
    with torch.no_grad():
        pred_probs = torch.sigmoid(pred_logits)
        pred_bin = (pred_probs > 0.5).float()
        
        # Dice & IoU
        intersection = (pred_bin * target).sum()
        union = pred_bin.sum() + target.sum()
        dice = (2. * intersection + 1e-5) / (union + 1e-5)
        iou = (intersection + 1e-5) / (union - intersection + 1e-5)
        
        # Accuracy (Pixel-wise)
        correct = (pred_bin == target).float().sum()
        accuracy = correct / target.numel()
        
    return dice.item(), iou.item(), accuracy.item()

# ================= 4. 训练与验证流程 =================
def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    epoch_loss = 0
    epoch_dice = 0
    epoch_iou = 0
    epoch_acc = 0
    
    loop = tqdm(loader, desc="Training", leave=False)
    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device)
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        dice, iou, acc = calculate_metrics(outputs, masks)
        epoch_loss += loss.item()
        epoch_dice += dice
        epoch_iou += iou
        epoch_acc += acc
        
        loop.set_postfix(loss=loss.item(), dice=dice, acc=acc)
        
    length = len(loader)
    return epoch_loss / length, epoch_dice / length, epoch_iou / length, epoch_acc / length

def validate(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_dice = 0
    epoch_iou = 0
    epoch_acc = 0
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device)
            masks = masks.to(device)
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            dice, iou, acc = calculate_metrics(outputs, masks)
            epoch_loss += loss.item()
            epoch_dice += dice
            epoch_iou += iou
            epoch_acc += acc
            
    length = len(loader)
    return epoch_loss / length, epoch_dice / length, epoch_iou / length, epoch_acc / length

# ================= 5. 绘图与保存功能 =================
def save_plots(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(15, 10))
    
    # Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.grid(True)
    
    # Dice
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['train_dice'], label='Train Dice')
    plt.plot(epochs, history['val_dice'], label='Val Dice')
    plt.title('Dice Coefficient')
    plt.legend()
    plt.grid(True)
    
    # IoU
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['train_iou'], label='Train IoU')
    plt.plot(epochs, history['val_iou'], label='Val IoU')
    plt.title('IoU')
    plt.legend()
    plt.grid(True)
    
    # Accuracy
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.title('Pixel Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ================= 主程序 =================
if __name__ == "__main__":
    print(f"[INFO] Using Device: {DEVICE}")
    print(f"[INFO] Logs will be saved to: {LOG_DIR}")
    
    # 加载数据
    try:
        all_images, all_masks = load_data_paths(DATA_PATH)
        print(f"[INFO] Found {len(all_images)} images.")
    except Exception as e:
        print(f"[ERROR] {e}")
        exit()
        
    if not all_images:
        print("[ERROR] No images found. Check path.")
        exit()

    train_x, valid_x, train_y, valid_y = train_test_split(
        all_images, all_masks, test_size=0.2, random_state=SEED
    )

    train_dataset = BUSIDataset(train_x, train_y, transform=train_transform)
    valid_dataset = BUSIDataset(valid_x, valid_y, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 模型构建
    print("[INFO] Building Model: UnetPlusPlus with EfficientNet-B7 encoder...")
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = BCEDiceLoss(bce_weight=0.5)
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # 历史记录字典
    history = {
        'epoch': [],
        'train_loss': [], 'val_loss': [],
        'train_dice': [], 'val_dice': [],
        'train_iou': [],  'val_iou': [],
        'train_acc': [],  'val_acc': []
    }

    best_dice = 0.0
    print("[INFO] Start Training...")
    
    for epoch in range(EPOCHS):
        # 训练与验证
        t_loss, t_dice, t_iou, t_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, DEVICE)
        v_loss, v_dice, v_iou, v_acc = validate(model, valid_loader, criterion, DEVICE)
        
        scheduler.step()
        
        # 更新历史记录
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_dice'].append(t_dice)
        history['val_dice'].append(v_dice)
        history['train_iou'].append(t_iou)
        history['val_iou'].append(v_iou)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)
        
        # 打印日志
        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
        print(f"    Train - Loss: {t_loss:.4f} | Dice: {t_dice:.4f} | IoU: {t_iou:.4f} | Acc: {t_acc:.4f}")
        print(f"    Valid - Loss: {v_loss:.4f} | Dice: {v_dice:.4f} | IoU: {v_iou:.4f} | Acc: {v_acc:.4f}")

        # 保存 CSV (每轮覆盖更新)
        df = pd.DataFrame(history)
        df.to_csv(CSV_SAVE_PATH, index=False)
        
        # 绘制并保存曲线图
        save_plots(history, PLOT_SAVE_PATH)

        # 保存最佳模型
        if v_dice > best_dice:
            print(f"    >>> Best Model Saved! (Dice improved: {best_dice:.4f} -> {v_dice:.4f})")
            best_dice = v_dice
            torch.save(model.state_dict(), WEIGHT_SAVE_PATH)
            
    print(f"[INFO] Training Complete. Best Weights: {WEIGHT_SAVE_PATH}")
    print(f"[INFO] Training Logs: {CSV_SAVE_PATH}")
    print(f"[INFO] Training Plots: {PLOT_SAVE_PATH}")