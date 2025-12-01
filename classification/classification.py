import os
import shutil
import time
import copy
import random
import numpy as np
import pandas as pd

# --- 关键修改：设置 Matplotlib 后端为 Agg，必须在导入 pyplot 之前 ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

# -------------------------------------------------
# 1. 配置与全局设置
# -------------------------------------------------
BASE_PATH = "/commondocument/group3"
RAW_DATA_DIR = os.path.join(BASE_PATH, "MuRAG-CAD/Dataset_BUSI_with_GT")
WORKING_DIR = os.path.join(BASE_PATH, "MuRAG-CAD/classification/workings")
LOG_DIR = os.path.join(BASE_PATH, "MuRAG-CAD/classification/logs")

os.makedirs(LOG_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(LOG_DIR, "DenseNet121_best.pth")
LOG_FILE_PATH = os.path.join(LOG_DIR, "training_log.txt")

def log_print(message):
    print(message)
    with open(LOG_FILE_PATH, "a") as f:
        f.write(message + "\n")

# 初始化日志
with open(LOG_FILE_PATH, "w") as f:
    f.write(f"Training Started at {time.ctime()}\n" + "="*50 + "\n")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_print(f"Using device: {device}")
log_print(f"Outputs will be saved directly to: {LOG_DIR}")

# -------------------------------------------------
# 2. 训练函数
# -------------------------------------------------
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=50, patience=15):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float('inf')
    consecutive_epochs_without_improvement = 0

    history = {'train_loss': [], 'validation_loss': [], 'train_acc': [], 'validation_acc': []}

    for epoch in range(num_epochs):
        log_print(f'Epoch {epoch+1}/{num_epochs}')
        log_print('-' * 10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            log_print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'validation':
                improved = False
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    log_print(f"--> New Best Accuracy: {best_acc:.4f} (Model Saved)")
                    improved = True
                
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    improved = True
                
                if improved:
                    consecutive_epochs_without_improvement = 0
                else:
                    consecutive_epochs_without_improvement += 1

        if consecutive_epochs_without_improvement >= patience:
            log_print(f"\nEarly stopping triggered. No improvement for {patience} epochs.")
            break
        log_print("") 

    time_elapsed = time.time() - since
    log_print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    log_print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model, history

# -------------------------------------------------
# 3. 主程序
# -------------------------------------------------
def main():
    # --- 数据清理 ---
    TRAIN_DIR = os.path.join(WORKING_DIR, "train")
    VAL_DIR = os.path.join(WORKING_DIR, "validation")
    TEST_DIR = os.path.join(WORKING_DIR, "test")

    if os.path.exists(WORKING_DIR):
        for item in os.listdir(WORKING_DIR):
            p = os.path.join(WORKING_DIR, item)
            if os.path.isfile(p): os.remove(p)
            else: shutil.rmtree(p)
    else:
        os.makedirs(WORKING_DIR)
    
    # --- 读取原始数据 ---
    file_paths = []
    labels = []
    if not os.path.exists(RAW_DATA_DIR):
        log_print(f"Error: Dataset path {RAW_DATA_DIR} does not exist.")
        return

    for label in os.listdir(RAW_DATA_DIR):
        label_dir = os.path.join(RAW_DATA_DIR, label)
        if os.path.isdir(label_dir):
            for f in os.listdir(label_dir):
                if f.endswith(".png") and "_mask" not in f:
                    file_paths.append(os.path.join(label_dir, f))
                    labels.append(label)

    df = pd.DataFrame({"Image_Path": file_paths, "Label": labels})
    
    # --- 划分数据集 ---
    train_val_df, test_df = train_test_split(df, test_size=0.10, stratify=df["Label"], random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.1112, stratify=train_val_df["Label"], random_state=42)

    for d, d_df in zip([TRAIN_DIR, VAL_DIR, TEST_DIR], [train_df, val_df, test_df]):
        for _, row in d_df.iterrows():
            dest_dir = os.path.join(d, row["Label"])
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(row["Image_Path"], dest_dir)

    # --- Transforms & Dataloaders ---
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ]),
        "validation": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ]),
        "test": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])
    }

    image_datasets = {x: ImageFolder(os.path.join(WORKING_DIR, x), transform=data_transforms[x])
                      for x in ["train", "validation", "test"]}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4)
                   for x in ["train", "validation", "test"]}
    dataset_sizes = {x: len(image_datasets[x]) for x in image_datasets}
    class_names = image_datasets["train"].classes

    # --- 类别权重计算 ---
    train_targets = image_datasets['train'].targets
    class_counts = torch.bincount(torch.tensor(train_targets))
    class_weights = 1. / torch.sqrt(class_counts.float())
    class_weights = class_weights / class_weights.sum() * len(class_names)
    class_weights = class_weights.to(device)

    # --- 模型与训练 ---
    log_print("Initializing DenseNet121...")
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, len(class_names))
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    model, history = train_model(
        model, criterion, optimizer, scheduler,
        dataloaders, dataset_sizes, device,
        num_epochs=60, patience=20
    )

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    log_print(f"Model saved to {MODEL_SAVE_PATH}")

    # ==========================================
    # 4. 指标保存与绘图 (无显示模式)
    # ==========================================

    # 4.1 保存训练历史 (CSV)
    history_df = pd.DataFrame(history)
    history_csv_path = os.path.join(LOG_DIR, "training_history.csv")
    history_df.to_csv(history_csv_path, index_label="epoch")
    log_print(f"Training history saved to {history_csv_path}")

    # 4.2 保存 Loss/Acc 曲线图 (PNG)
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['validation_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc', linewidth=2)
    plt.plot(history['validation_acc'], label='Validation Acc', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True)

    curve_path = os.path.join(LOG_DIR, "loss_acc_curves.png")
    plt.tight_layout()
    plt.savefig(curve_path, dpi=300)
    plt.close() # 显式关闭，释放内存
    log_print(f"Loss/Acc curves saved to {curve_path}")

    # 4.3 测试集推理
    log_print("Evaluating on Test Set...")
    model.eval()
    y_true = []
    y_pred = []
    y_probs = []

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # 4.4 保存分类报告 (CSV)
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_csv_path = os.path.join(LOG_DIR, "classification_report.csv")
    report_df.to_csv(report_csv_path)
    log_print(f"Classification report saved to {report_csv_path}")

    # 4.5 保存混淆矩阵 (CSV + PNG)
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(LOG_DIR, "confusion_matrix.csv"))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    cm_path = os.path.join(LOG_DIR, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    log_print(f"Confusion Matrix saved to {cm_path}")

    # 4.6 保存 ROC 曲线与 AUC (CSV + PNG)
    n_classes = len(class_names)
    if n_classes >= 2:
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(10, 8))
        colors = cycle(['blue', 'red', 'green', 'orange', 'purple'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC curve of {class_names[i]} (area = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class ROC')
        plt.legend(loc="lower right")
        
        roc_path = os.path.join(LOG_DIR, "roc_curve.png")
        plt.savefig(roc_path, dpi=300)
        plt.close()
        log_print(f"ROC Curve saved to {roc_path}")
        
        auc_df = pd.DataFrame({'Class': class_names, 'AUC': list(roc_auc.values())})
        auc_df.to_csv(os.path.join(LOG_DIR, "auc_scores.csv"), index=False)

    log_print("\nAll metrics and plots have been saved successfully.")

if __name__ == "__main__":
    main()