import streamlit as st
import subprocess
import os
import time
import re
import html
from PIL import Image

# ================= 1. 配置区域 (Configuration) =================

# 脚本路径 (需确保该脚本有执行权限: chmod +x run_pipeline.sh)
SHELL_SCRIPT_PATH = "./run_pipeline.sh"

# 基础路径 (根据你的实际服务器路径修改)
# 如果你是按照 README 克隆的，通常是当前目录
BASE_DIR = os.getcwd() 

# 结果输出路径 (需与 run_pipeline.sh 中的输出一致)
CLS_RESULT_DIR = os.path.join(BASE_DIR, "classification/single_results")
SEG_RESULT_DIR = os.path.join(BASE_DIR, "segmentation/signal_results")

# 页面基础设置
st.set_page_config(page_title="MuRAG-CAD 智能诊断系统", layout="wide", page_icon="🩺")

# ================= 2. 自定义 CSS (UI 美化) =================
st.markdown("""
<style>
    /* 全局字体优化 - 使用微软雅黑/系统字体 */
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
    }
    
    .diagnosis-title { 
        font-size: 20px !important; 
        color: #666; 
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 10px; 
    }
    .diagnosis-result { 
        font-size: 38px !important; 
        font-weight: 900;      
        margin: 0;
        line-height: 1.3;
    }
    
    /* 状态颜色 */
    .malignant { background: linear-gradient(135deg, #fff5f5 0%, #ffebee 100%); border-color: #ffcdd2; color: #d32f2f; }
    .benign { background: linear-gradient(135deg, #f1f8e9 0%, #e8f5e9 100%); border-color: #c8e6c9; color: #2e7d32; }
    .normal { background: linear-gradient(135deg, #e3f2fd 0%, #e1f5fe 100%); border-color: #b3e5fc; color: #0277bd; }
    .pending { background-color: #f8f9fa; border-color: #e9ecef; color: #adb5bd; }

    /* --- 终端日志窗口 (仿 VSCode 风格) --- */
    .terminal-container {
        border-radius: 8px;
        background-color: #1e1e1e; /* 深灰背景 */
        border: 1px solid #333;
        padding: 0; 
        margin-bottom: 20px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .terminal-header {
        background-color: #252526;
        padding: 8px 15px;
        color: #cccccc;
        font-size: 13px;
        border-bottom: 1px solid #333;
        font-family: 'Consolas', monospace;
        display: flex;
        justify-content: space-between;
    }
    .terminal-window {
        padding: 15px;
        color: #d4d4d4;
        font-family: 'Consolas', 'Courier New', monospace !important;
        font-size: 16px !important; /* 字体加大 */
        line-height: 1.6;
        height: 400px;             
        overflow-y: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    /* 自定义滚动条 */
    .terminal-window::-webkit-scrollbar { width: 10px; }
    .terminal-window::-webkit-scrollbar-track { background: #1e1e1e; }
    .terminal-window::-webkit-scrollbar-thumb { background: #444; border-radius: 5px; }
    .terminal-window::-webkit-scrollbar-thumb:hover { background: #555; }

    /* 日志高亮颜色 */
    .log-info { color: #4ec9b0; font-weight: 600; }       /* 青色 */
    .log-warn { color: #dcdcaa; }       /* 黄色 */
    .log-error { color: #f44747; font-weight: bold; } /* 红色 */
    .log-dim { color: #808080; font-style: italic;}        /* 灰色 */
    .log-highlight { color: #569cd6; font-weight: bold; font-size: 18px; display: block; margin-top: 10px; border-bottom: 1px dashed #444;} /* 蓝色标题 */
    
    .img-label {
        text-align: center;
        font-weight: 700;
        font-size: 18px !important;
        margin-top: 15px;
        margin-bottom: 8px;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# ================= 3. 辅助函数 =================

def get_result_paths(input_path):
    """根据输入图片路径，推断生成结果的路径"""
    filename = os.path.basename(input_path)
    # 假设输入是 image.png，去除后缀得到 image
    name_no_ext = os.path.splitext(filename)[0]
    
    # 1. 分割结果图 (step2_seg.py 生成 _prediction.png)
    seg_img_path = os.path.join(SEG_RESULT_DIR, f"{name_no_ext}_prediction.png")
    
    # 2. 分类热力图 (step1_cls.py 生成 _heatmap.png 或 _result.png)
    # 注意：这里假设你的脚本生成的是 _result.png 或 _heatmap.png，请根据实际情况调整
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
        pred_match = re.search(r"(Prediction|最终预测):\s*(\w+)", content, re.IGNORECASE)
        if not pred_match: 
            return "解析失败", "pending", {}
            
        pred_class = pred_match.group(2).strip().lower()
        
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
    """格式化日志行，添加 HTML 颜色，过滤垃圾日志"""
    # 1. 清理乱码和格式
    line = line.replace("**", "").strip()
    # 过滤掉底层库的繁杂日志
    ignore_keywords = [
        "oneapi", "avx", "cuda", "successfully opened", 
        "skipping register", "check_version.py", "fetch_version_info",
        "dso_loader", "numa", "stream_executor"
    ]
    
    safe_line = html.escape(line)
    if not safe_line: return None
    lower_line = safe_line.lower()

    if any(k in lower_line for k in ignore_keywords): return None

    # 2. 颜色规则
    if "error" in lower_line or "exception" in lower_line or "traceback" in lower_line:
        return f'<span class="log-error">❌ {safe_line}</span>'
    elif "warning" in lower_line:
        return f'<span class="log-warn">⚠️ {safe_line}</span>'
    elif any(k in lower_line for k in ["processing", "loading", "running", "analyzing"]):
        return f'<span class="log-dim">>> {safe_line}</span>'
    elif any(k in lower_line for k in ["success", "saved", "completed", "done", "prediction:"]):
        return f'<span class="log-info">✅ {safe_line}</span>'
    # 识别阶段标题 (Key Stages)
    elif any(k in lower_line for k in ["step 1", "step 2", "step 3", "classification", "segmentation", "report", "rag"]):
        return f'<br><span class="log-highlight">🔹 {safe_line}</span>'
    else:
        return safe_line

# ================= 4. 主界面逻辑 =================

st.title("🩺 MuRAG-CAD: Multimodal Breast Ultrasound Diagnosis")
st.markdown("集成 **DenseNet121** (分类) + **Unet++** (分割) + **DeepSeek-RAG** (报告生成)")

# --- 输入栏 ---
with st.container():
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        # 默认路径 (为了演示方便)
        # 建议修改为项目内的测试图片路径
        default_path = os.path.join(BASE_DIR, "classification/workings/test/malignant/malignant (39).png") 
        if not os.path.exists(default_path): default_path = "" # 如果不存在则置空
        
        img_path = st.text_input("📁 请输入超声影像文件路径 (Abs Path):", value=default_path)
        
    with col_btn:
        st.write("") 
        st.write("") 
        start_btn = st.button("🚀 开始诊断", type="primary", use_container_width=True)

# --- 占位符初始化 (避免页面跳动) ---
result_placeholder = st.empty()       
image_gallery_placeholder = st.empty() 
status_placeholder = st.empty()       
log_placeholder = st.empty()

# 初始状态显示
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
        st.error(f"❌ 找不到执行脚本: {SHELL_SCRIPT_PATH}")
    else:
        # 准备工作
        log_lines = []
        seg_img, cls_img, cls_report_file = get_result_paths(img_path)
        
        # 进度状态容器
        with status_placeholder.status("⚙️ 系统正在全速运行...", expanded=True) as status:
            step_text = st.empty()
            progress_bar = st.progress(0)
            
            # 初始化日志窗口
            log_placeholder.markdown(
                '<div class="terminal-container"><div class="terminal-header"><span>Output Terminal</span><span>bash run_pipeline.sh</span></div><div class="terminal-window">Initializing environment...</div></div>', 
                unsafe_allow_html=True
            )
            
            step_text.markdown("#### ⏳ 正在初始化环境...")
            
            # === 阶段定义 (用于控制进度条和演示延迟) ===
            STAGES = { "init": 0, "cls": 1, "seg": 2, "rag": 3 }
            current_stage = 0
            last_log_update = 0

            try:
                # 启动 Shell 脚本
                env_vars = os.environ.copy()
                env_vars["PYTHONUNBUFFERED"] = "1" # 强制 Python 不缓存输出

                process = subprocess.Popen(
                    ["bash", SHELL_SCRIPT_PATH, img_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,     
                    env=env_vars   
                )
                
                # --- 实时读取日志循环 ---
                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    
                    if line:
                        fmt_line = format_log_line(line)
                        if fmt_line:
                            log_lines.append(fmt_line)
                            
                            # 控制刷新频率，避免前端卡死
                            if time.time() - last_log_update > 0.1 or "step" in line.lower():
                                display_logs = log_lines # 显示全部日志
                                log_html = "<br>".join(display_logs)
                                log_placeholder.markdown(
                                    f'<div class="terminal-container"><div class="terminal-header"><span>Output Terminal</span><span>Processing...</span></div><div class="terminal-window" id="log-window">{log_html}</div></div>', 
                                    unsafe_allow_html=True
                                )
                                last_log_update = time.time()
                        
                        # === 智能进度更新 & 演示延迟逻辑 ===
                        lower = line.lower()
                        new_stage = current_stage

                        # 1. 监测到分类阶段 (DenseNet)
                        if ("classification" in lower or "step 1" in lower) and current_stage < STAGES["cls"]:
                            new_stage = STAGES["cls"]
                            step_text.markdown("#### 🔍 阶段 1/3: 智能分类 (DenseNet121)...")
                            progress_bar.progress(30)
                            # 演示延迟: 3秒
                            time.sleep(3)

                        # 2. 监测到分割阶段 (Unet++)
                        elif ("segmentation" in lower or "step 2" in lower) and current_stage < STAGES["seg"]:
                            new_stage = STAGES["seg"]
                            step_text.markdown("#### ✂️ 阶段 2/3: 病灶分割 (Unet++)...")
                            progress_bar.progress(60)
                            # 演示延迟: 6秒 (分割通常比较慢)
                            time.sleep(6)

                        # 3. 监测到报告生成阶段 (RAG)
                        elif ("deepseek" in lower or "report" in lower or "rag" in lower) and current_stage < STAGES["rag"]:
                            new_stage = STAGES["rag"]
                            step_text.markdown("#### 📝 阶段 3/3: 生成诊断报告 (DeepSeek-RAG)...")
                            progress_bar.progress(90)
                            # 演示延迟: 2秒
                            time.sleep(2)
                        
                        current_stage = new_stage

                process.wait()
                
                # 最后一次刷新日志
                log_html = "<br>".join(log_lines)
                log_placeholder.markdown(
                    f'<div class="terminal-container"><div class="terminal-header"><span>Output Terminal</span><span>Done</span></div><div class="terminal-window" id="log-window">{log_html}</div></div>', 
                    unsafe_allow_html=True
                )
                
                if process.returncode == 0:
                    progress_bar.progress(100)
                    step_text.markdown("#### ✅ 分析完成！")
                    time.sleep(0.5)
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

        # 2. 图像展示区 (左右两栏)
        with image_gallery_placeholder.container():
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="img-label">🔥 可解释性热力图 (Grad-CAM)</div>', unsafe_allow_html=True)
                if os.path.exists(cls_img):
                    st.image(Image.open(cls_img), use_container_width=True)
                else:
                    st.warning("未找到热力图结果")
            
            with col2:
                st.markdown('<div class="img-label">🎯 病灶分割预测 (Segmentation)</div>', unsafe_allow_html=True)
                if os.path.exists(seg_img):
                    st.image(Image.open(seg_img), use_container_width=True)
                else:
                    st.warning("未找到分割结果")

            # 3. 详细报告展开
            st.markdown("---")
            with st.expander("📄 查看完整 AI 诊断报告 (点击展开)", expanded=False):
                if raw_report_content:
                    st.text_area("LLM 生成报告", raw_report_content, height=300)
                else:
                    st.info("暂无文本报告数据")