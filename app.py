import streamlit as st
import subprocess
import os
import time
import re
import html
from PIL import Image

# ================= 1. 配置区域 (请根据你的实际路径修改) =================
# Shell 脚本路径
SHELL_SCRIPT_PATH = "./run_pipeline.sh"

# 结果保存的基础路径 (需与 Shell 脚本中的 OUT_DIR 保持一致)
BASE_OUTPUT_DIR = "/commondocument/group3/MuRAG-CAD"
CLS_RESULT_DIR = os.path.join(BASE_OUTPUT_DIR, "classification/single_results")
SEG_RESULT_DIR = os.path.join(BASE_OUTPUT_DIR, "segmentation/signal_results")

# 页面设置
st.set_page_config(page_title="AI 乳腺癌智能诊断系统", layout="wide", page_icon="🩺")

# ================= 2. 自定义 CSS (美化界面) =================
st.markdown("""
<style>
    /* 全局字体优化 */
    html, body, [class*="css"] {
        font-family: "Microsoft YaHei", "Segoe UI", sans-serif !important;
    }

    /* 诊断结果卡片样式 */
    .diagnosis-card {
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid #e0e0e0;
        background-color: #ffffff;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: transform 0.3s ease;
    }
    .diagnosis-card:hover { transform: translateY(-2px); }
    
    .diagnosis-title { 
        font-size: 18px !important; 
        color: #666; 
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px; 
    }
    .diagnosis-result { 
        font-size: 36px !important; 
        font-weight: 900;      
        margin: 0;
        line-height: 1.2;
    }
    
    /* 状态颜色 */
    .malignant { background: linear-gradient(135deg, #fff5f5 0%, #ffebee 100%); border-color: #ffcdd2; color: #d32f2f; }
    .benign { background: linear-gradient(135deg, #f1f8e9 0%, #e8f5e9 100%); border-color: #c8e6c9; color: #2e7d32; }
    .normal { background: linear-gradient(135deg, #e3f2fd 0%, #e1f5fe 100%); border-color: #b3e5fc; color: #0277bd; }
    .pending { background-color: #f8f9fa; border-color: #e9ecef; color: #adb5bd; }

    /* --- 终端日志窗口 --- */
    .terminal-container {
        border-radius: 8px;
        background-color: #0e1117; /* 深色背景 */
        border: 1px solid #30363d;
        padding: 0; 
        margin-bottom: 20px;
        overflow: hidden;
    }
    .terminal-header {
        background-color: #21262d;
        padding: 5px 15px;
        color: #8b949e;
        font-size: 12px;
        border-bottom: 1px solid #30363d;
        font-family: monospace;
    }
    .terminal-window {
        padding: 15px;
        color: #c9d1d9;
        font-family: 'Consolas', 'Courier New', monospace !important;
        font-size: 14px !important;
        line-height: 1.6;
        height: 350px;             
        overflow-y: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    /* 自定义滚动条 */
    .terminal-window::-webkit-scrollbar { width: 8px; }
    .terminal-window::-webkit-scrollbar-track { background: #0e1117; }
    .terminal-window::-webkit-scrollbar-thumb { background: #30363d; border-radius: 4px; }

    /* 日志高亮颜色 */
    .log-info { color: #56d364; }       /* 绿色 */
    .log-warn { color: #d29922; }       /* 黄色 */
    .log-error { color: #f85149; font-weight: bold; } /* 红色 */
    .log-dim { color: #8b949e; }        /* 灰色 */
    .log-highlight { color: #58a6ff; font-weight: bold; font-size: 15px; margin-top: 10px; display: block; border-bottom: 1px solid #30363d; padding-bottom: 2px;} /* 蓝色标题 */
    
    .img-label {
        text-align: center;
        font-weight: 600;
        font-size: 16px !important;
        margin-top: 12px;
        margin-bottom: 5px;
        color: #444;
    }
</style>
""", unsafe_allow_html=True)

# ================= 3. 辅助函数 =================

def get_result_paths(input_path):
    """根据输入图片路径，推断生成结果的路径"""
    filename = os.path.basename(input_path)
    # Python 逻辑是 split('.')[0]，例如 'case.01.png' -> 'case'
    name_no_ext = filename.split('.')[0] 
    
    # 1. 分割结果图 (step2_seg.py 生成 _prediction.png)
    seg_img_path = os.path.join(SEG_RESULT_DIR, f"{name_no_ext}_prediction.png")
    
    # 2. 分类结果图 (step1_cls.py 生成 _result.png)
    cls_img_path = os.path.join(CLS_RESULT_DIR, f"{name_no_ext}_result.png")
    
    # 3. 分类报告文本
    report_path = os.path.join(CLS_RESULT_DIR, f"{name_no_ext}_report.txt")
    
    return seg_img_path, cls_img_path, report_path

def parse_classification_report(report_path):
    """解析 step1_cls.py 生成的 report.txt"""
    if not os.path.exists(report_path): 
        return "等待分析...", "pending", {}
    
    try:
        with open(report_path, 'r', encoding='utf-8') as f: 
            content = f.read()
        
        # 匹配 Prediction: Malignant
        pred_match = re.search(r"Prediction:\s*(\w+)", content, re.IGNORECASE)
        if not pred_match: 
            return "解析失败", "pending", {}
            
        pred_class = pred_match.group(1).strip().lower()
        
        # 匹配 Probabilities (例如 malignant: 98.50%)
        prob_match = re.search(rf"{pred_class}\s*:\s*([\d\.]+)%", content, re.IGNORECASE)
        probability = prob_match.group(1) if prob_match else "N/A"
        
        cn_map = { "malignant": "恶性 (Malignant)", "benign": "良性 (Benign)", "normal": "正常 (Normal)" }
        style_map = { "malignant": "malignant", "benign": "benign", "normal": "normal" }
        
        display_text = f"{cn_map.get(pred_class, pred_class.upper())} <br><span style='font-size:20px; color:#666;'>置信度: {probability}%</span>"
        
        return display_text, style_map.get(pred_class, "pending"), content
    except Exception as e:
        return f"读取错误: {str(e)}", "pending", {}

def format_log_line(line):
    """格式化日志行，添加 HTML 颜色"""
    safe_line = html.escape(line.strip())
    if not safe_line: return ""
    lower_line = safe_line.lower()
    
    # 过滤掉一些无关紧要的底层库日志
    ignore_keywords = ["oneapi", "avx", "cuda", "successfully opened", "skipping register","check_version.py", "fetch_version_info"]
    if any(k in lower_line for k in ignore_keywords): return None

    if "error" in lower_line or "exception" in lower_line or "traceback" in lower_line:
        return f'<span class="log-error">❌ {safe_line}</span>'
    elif "warning" in lower_line:
        return f'<span class="log-warn">⚠️ {safe_line}</span>'
    elif any(k in lower_line for k in ["running", "loading", "analyzing", "processing"]):
        return f'<span class="log-dim">>> {safe_line}</span>'
    elif any(k in lower_line for k in ["success", "saved", "completed", "done", "prediction:"]):
        return f'<span class="log-info">✅ {safe_line}</span>'
    # 识别 Shell 脚本中的阶段标题
    elif "----------" in lower_line or "step" in lower_line:
        return f'<span class="log-highlight">🔹 {safe_line}</span>'
    else:
        return safe_line

# ================= 4. 主界面逻辑 =================

st.title("🩺 AI 乳腺癌多模态辅助诊断系统")
st.markdown("集成 **DenseNet121** (分类) + **Unet++** (分割) + **DeepSeek-RAG** (报告生成)")

# 输入栏
with st.container():
    col_input, col_btn = st.columns([4, 1])
    with col_input:
        # 默认路径方便测试
        default_path = "/commondocument/group3/MuRAG-CAD/classification/workings/test/malignant/malignant (39).png"
        img_path = st.text_input("📁 请输入超声影像文件路径:", value=default_path)
    with col_btn:
        st.write("") 
        st.write("") 
        start_btn = st.button("🚀 开始一键诊断", type="primary", use_container_width=True)

# 占位符初始化
result_placeholder = st.empty()       
image_gallery_placeholder = st.empty() 
status_placeholder = st.empty()       
log_placeholder = st.empty()

# 初始显示等待卡片
with result_placeholder.container():
    st.markdown("""
        <div class="diagnosis-card pending">
            <div class="diagnosis-title">SYSTEM STATUS</div>
            <div class="diagnosis-result">READY TO ANALYZE</div>
        </div>
    """, unsafe_allow_html=True)

if start_btn:
    if not os.path.exists(img_path):
        st.error(f"❌ 文件不存在: {img_path}")
    elif not os.path.exists(SHELL_SCRIPT_PATH):
        st.error(f"❌ 找不到执行脚本: {SHELL_SCRIPT_PATH}，请确保脚本在当前目录下。")
    else:
        # 清空之前的状态
        log_lines = []
        seg_img, cls_img, cls_report_file = get_result_paths(img_path)
        
        # 进度状态容器
        with status_placeholder.status("⚙️ 系统正在全速运行...", expanded=True) as status:
            step_text = st.empty()
            progress_bar = st.progress(0)
            
            # 初始化日志窗口
            log_placeholder.markdown(
                '<div class="terminal-container"><div class="terminal-header">Terminal Output</div><div class="terminal-window">Initializing...</div></div>', 
                unsafe_allow_html=True
            )
            
            step_text.markdown("#### ⏳ 正在初始化环境...")
            
            # 阶段标记，用于进度条逻辑
            STAGES = { "init": 0, "cls": 1, "seg": 2, "rag": 3 }
            current_stage = 0

            try:
                # 启动 Shell 脚本
                # use unbuffered output
                env_vars = os.environ.copy()
                env_vars["PYTHONUNBUFFERED"] = "1"

                process = subprocess.Popen(
                    ["bash", SHELL_SCRIPT_PATH, img_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,     
                    env=env_vars   
                )
                
                # 实时读取日志
                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    
                    if line:
                        fmt_line = format_log_line(line)
                        if fmt_line:
                            log_lines.append(fmt_line)
                            
                            # 更新日志窗口 (只显示最近 50 行防止卡顿)
                            display_logs = log_lines # [-50:] 
                            log_html = "<br>".join(display_logs)
                            log_placeholder.markdown(
                                f'<div class="terminal-container"><div class="terminal-header">Running: {os.path.basename(SHELL_SCRIPT_PATH)}</div><div class="terminal-window" id="log-window">{log_html}<br><span id="anchor"></span></div></div>', 
                                unsafe_allow_html=True
                            )
                        
                        # === 智能进度更新逻辑 ===
                        lower = line.lower()
                        
                        # 阶段 1: 分类
                        if "classification" in lower and "running" in lower and current_stage < STAGES["cls"]:
                            current_stage = STAGES["cls"]
                            step_text.markdown("#### 🔍 阶段 1/3: 智能分类 (DenseNet121)...")
                            progress_bar.progress(20)
                            time.sleep(1) # 演示用延迟，增加仪式感

                        # 阶段 2: 分割
                        elif "segmentation" in lower and "running" in lower and current_stage < STAGES["seg"]:
                            current_stage = STAGES["seg"]
                            step_text.markdown("#### ✂️ 阶段 2/3: 病灶分割 (Unet++)...")
                            progress_bar.progress(55)
                            time.sleep(2) # 演示用，因为分割通常比较慢

                        # 阶段 3: 报告
                        elif "deepseek" in lower and "report" in lower and current_stage < STAGES["rag"]:
                            current_stage = STAGES["rag"]
                            step_text.markdown("#### 📝 阶段 3/3: 生成诊断报告 (DeepSeek-RAG)...")
                            progress_bar.progress(85)
                            time.sleep(1)

                process.wait()
                
                if process.returncode == 0:
                    progress_bar.progress(100)
                    step_text.markdown("#### ✅ 分析完成！")
                    status.update(label="✅ 分析完成", state="complete", expanded=False)
                else:
                    status.update(label="❌ 分析出错", state="error")
                    st.error("脚本执行返回错误代码，请检查日志。")

            except Exception as e:
                st.error(f"执行异常: {str(e)}")

        # === 结果展示 ===
        diag_text, diag_style, raw_report_content = parse_classification_report(cls_report_file)

        # 1. 顶部大卡片
        with result_placeholder.container():
            st.markdown(f"""
                <div class="diagnosis-card {diag_style}">
                    <div class="diagnosis-title">AI DIAGNOSIS RESULT</div>
                    <div class="diagnosis-result">{diag_text}</div>
                </div>
            """, unsafe_allow_html=True)

        # 2. 图像展示区
        with image_gallery_placeholder.container():
            st.subheader("🖼️ 影像分析详情")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="img-label">🔥 可解释性热力图 (Grad-CAM)</div>', unsafe_allow_html=True)
                if os.path.exists(cls_img):
                    # cls_img 是分类脚本生成的包含原图+热力图的对比图
                    st.image(Image.open(cls_img), use_container_width=True)
                else:
                    st.warning("未找到热力图结果")
            
            with col2:
                st.markdown('<div class="img-label">🎯 病灶分割预测 (Segmentation)</div>', unsafe_allow_html=True)
                if os.path.exists(seg_img):
                    # seg_img 是分割脚本生成的包含原图+Mask的对比图
                    st.image(Image.open(seg_img), use_container_width=True)
                else:
                    st.warning("未找到分割结果")

            # 3. 详细报告展开
            st.markdown("---")
            with st.expander("📄 查看完整 AI 诊断报告数据 (点击展开)", expanded=False):
                if raw_report_content:
                    st.text_area("分类报告原始数据", raw_report_content, height=200)
                else:
                    st.info("暂无文本报告数据")
                
                # 尝试读取分割报告
                seg_txt_path = cls_report_file.replace("classification", "segmentation").replace("single_results", "signal_results")
                if os.path.exists(seg_txt_path):
                     with open(seg_txt_path, 'r', encoding='utf-8') as f:
                         st.text_area("分割测量原始数据", f.read(), height=200)