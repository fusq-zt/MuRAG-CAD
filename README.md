# 🩺 MuRAG-CAD: Multimodal RAG Breast Ultrasound Diagnosis System

MuRAG-CAD 是一个集成深度学习 CAD（计算机辅助诊断）与大型语言模型 (LLM) 的乳腺超声智能诊断系统。项目参考了ChatCAD的实现
本项目结合了：

* **DenseNet121** — 乳腺良恶性分类
* **Unet++（EfficientNet-B7 Encoder）** — 病灶分割
* **DeepSeek + LangChain RAG** — 自动报告生成
* **Streamlit Web UI** — 可视化诊断界面

旨在生成完整可解释的医学诊断流程。如果此项目帮助到您，帮我点个⭐
<img width="1911" height="1773" alt="Untitled diagram-2025-12-01-080858" src="https://github.com/user-attachments/assets/f8a9b10f-59b7-4c75-b8d7-6b9de5ee52ec" />

---

## ✨ 核心功能 (Features)

### 🔍 良恶性分类（DenseNet121）

* 自动判断乳腺超声图像良恶性(Acc = 0.94)
* 生成 Grad-CAM 热力图提升可解释性

### ✂️ 病灶精准分割（Unet++）

* 使用 EfficientNet-B7 编码器
* 自动勾勒病灶区域（Dice = 0.81）
* 计算几何特征（面积、长径、位置等）

### 📝 RAG 自动诊断报告

* DeepSeek LLM + LangChain 检索增强
* 结合临床指南生成：

  * 影像学描述
  * BI-RADS 分类
  * 诊断建议

### 🖥️ Streamlit 可视化界面

* 一键诊断
* 实时日志
* 可视化对比展示

---
## 复现该项目

## 🛠️ 环境安装 (Installation)

项目提供 Conda 环境文件 `environment.yml`，可一键安装：

```bash
# 1. 克隆项目
git clone https://github.com/YourUsername/MuRAG-CAD.git
cd MuRAG-CAD

# 2. 使用 .yml 文件创建环境
conda env create -f environment.yml

# 3. 激活环境
conda activate murag-cad
```
---

## 📥 数据与权重放置说明 (Setup)

⚠️ 仓库不包含模型权重和大数据集，请按以下步骤准备：

### 1️⃣ 数据集准备 (Dataset Preparation)

* 本项目使用 **BUSI (Breast Ultrasound Images)** 数据集进行训练。
* 下载 BUSI 数据集（可从 Kaggle 或官方渠道获取）。
* 解压并将文件夹重命名为 `Dataset_BUSI_with_GT`，放置在项目根目录下。

目录结构示例：

```
MuRAG-CAD/
└── Dataset_BUSI_with_GT/
    ├── benign/       # 良性样本
    ├── malignant/    # 恶性样本
    └── normal/       # 正常样本
```

### 2️⃣ 模型权重放置 (Model Weights)

训练完成后，将最佳权重放入以下路径：
（训练代码为MuRAG-CAD/classification/classification.py，MuRAG-CAD/segmentation/seg.py
训练后可以在对应logs文件夹下找到模型训练文件)

| 模型   | 文件名                  | 推荐路径                 |
| ---- | -------------------- | -------------------- |
| 分类模型 | DenseNet121_best.pth | classification/logs/ |
| 分割模型 | best_unet.pth        | segmentation/logs/   |



### 3️⃣ 测试数据 (Test Data)
---

## 🔑 API 配置（DeepSeek LLM）

项目依赖 DeepSeek API 进行报告生成。

设置环境变量（Linux/Mac）：

```bash
export DEEPSEEK_API_KEY="sk-your-key"
```
或在 `rag/deepseek_pdf.py` 中直接修改。

---

## 🚀 快速开始 (Quick Start)

### ✅ 方式一：使用 Streamlit Web 界面（推荐）

```bash
streamlit run app.py
```

浏览器访问：

```
http://localhost:8501
```
在页面上输入超声图像路径 → 点击“开始一键诊断”。

<img width="2945" height="704" alt="image" src="https://github.com/user-attachments/assets/0b4251d7-b31b-4ce6-bf28-c7f06918f26c" />
<img width="2917" height="1384" alt="image" src="https://github.com/user-attachments/assets/f9539202-1675-4fd5-810f-6c7641446f6e" />
<img width="2422" height="982" alt="image" src="https://github.com/user-attachments/assets/26ebd9c6-5931-41e1-9955-b5dcc6a6bd24" />


### ✅ 方式二：命令行脚本运行

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh "/path/to/image.png"
```

---


## 📂 项目结构 (Project Structure)

```
MuRAG-CAD/
├── Dataset_BUSI_with_GT/   <-- [核心数据集]
│   ├── benign/
│   ├── malignant/
│   └── normal/
├── classification/
│   ├── train.py
│   ├── predict_single_image.py
│   └── logs/               # [存放分类权重]
├── segmentation/
│   ├── train.py
│   ├── predict_single_image.py
│   └── logs/               # [存放分割权重]
├── rag/
│   └── deepseek_pdf.py     # 报告生成（LLM + RAG）
├── chroma_db/              # RAG 向量库 (自动生成)
├── app.py                  # Streamlit 前端入口
├── run_pipeline.sh         # 全流程脚本
└── environment.yml         # Conda 环境文件
```
---

## 🤝 贡献 (Contributing)

欢迎提交：
* Bug 反馈
* 新功能建议
* Pull Requests
---
