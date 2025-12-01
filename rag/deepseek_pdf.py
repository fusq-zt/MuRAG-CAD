# -*- coding: utf-8 -*-
import sys
import io
import os
import torch
import argparse  # 新增：用于解析命令行参数
# from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# 全局中文输出支持
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ================= 配置区域 =================
from env_utils import DEEPSEEK_API_KEY
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


# 默认路径（如果运行时不传参数，则使用这些路径）

DEFAULT_CLS_PATH = "MuRAG-CAD/classification/single_results/malignant (39)_report.txt"
DEFAULT_DET_PATH = "MuRAG-CAD/segmentation/signal_results/malignant (39)_report.txt"

def get_vector_store():
    """获取向量数据库实例"""
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-zh-v1.5",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    # 若无本地数据库，此处需根据实际情况调整（如创建新库或跳过）
    vectorstore = Chroma(
        collection_name='medical_pdf_db', 
        embedding_function=embeddings,
        persist_directory='./chroma_db'
    )
    return vectorstore

def load_file_content(filepath):
    """读取指定路径的txt文件内容"""
    if not os.path.exists(filepath):
        print(f"❌ [错误] 找不到文件: {filepath}")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"✅ [成功] 已加载: {os.path.basename(filepath)}")
            return content
    except Exception as e:
        print(f"❌ [读取异常] {filepath}: {e}")
        return None

def ask_breast_analysis(llm, combined_report_text):
    """
    核心生成函数：影像分析 + 临床查体 + 诊断建议
    """
    # 1. 尝试检索指南
    try:
        vectorstore = get_vector_store()
        similar_docs = vectorstore.similarity_search(combined_report_text, k=1)
        rag_context = "\n".join([doc.page_content for doc in similar_docs]) if similar_docs else "无外部指南上下文。"
    except Exception:
        rag_context = "（向量数据库未连接，仅基于输入数据分析）"

    # ================= 核心 Prompt =================
    template_text = """
    你是一名辅助诊断的AI助手。请基于【输入数据】和【参考指南】，完成三部分报告。

    【输入数据 - AI检测报告】：
    {question}

    【参考指南】：
    {context}

    --------------------------------------------------
    【第一部分：影像学表现分析】
    
    请提取文本中的关键数据，仅描述客观影像事实：

    1. **基本信息**：分行列出图片名称、AI预测类别（Benign/Malignant）及具体概率值。
    2. **病灶位置分析**：分行根据【中心坐标 (Center)】相对于【图片尺寸】的位置，分析病灶位于视野的什么方位（如：左上、正中、偏下等）。
    3. **病灶大小与形态**：分行列出【边界框尺寸 (Bounding Box)】和【像素面积】。简单分析病灶的相对大小（例如：是微小结节，还是占据图像大部分的巨大团块）。

    *注意：影像学表现只分析文本中存在的数据（坐标、尺寸、概率），不要编造回声、血流等未提及的信息。*

    --------------------------------------------------
    【第二部分：通用乳腺临床查体报告】
    
    请使用以下**严格固定的模板**。根据 AI 数据的**良恶性**和**像素大小**来填充括号内的内容，不要随意发挥，保持临床病历的简洁性。
    
    *填充逻辑*：
    1. **皮肤情况**：
       - 恶性且巨大 -> 填"可见局部隆起"或"呈橘皮样变"。
       - 良性或小病灶 -> 填"无红肿、破溃"。
    2. **腺体质地/触痛/肿物**：
       - Malignant (恶性) -> 填"质硬"、"无压痛"、"可触及质硬肿物，边界不清"。
       - Benign (良性) -> 填"软"或"质韧"、"无压痛"、"未扪及明显肿物"（若AI框很小）或"可触及类圆形结节"（若AI框较大）。
    3. **淋巴结**：
       - 恶性概率>95% -> 填"可触及肿大"。
       - 其他 -> 填"无肿大"。
    4. **临床诊断**：
        结合参考指南进行判断
       - 良性 -> "右/左乳肿物" 或 "乳腺增生"等
       - 恶性 -> "右/左乳肿物（性质待查）"等

    【目标模板】：
    通用乳腺临床查体报告
    【查体记录】
    双侧乳房对称等大，发育正常，表面皮肤[1. 皮肤情况]。[2. 特殊外观]（默认为空，除非AI暗示有术后痕迹）。
    双侧乳头平齐，无偏斜、糜烂，无乳头凹陷。挤压双侧乳头，[3. 溢液情况]（默认：未见明显溢血溢液）。
    双侧乳房触诊腺体[4. 腺体质地]，[5. 触痛情况]，[6. 肿物情况]。
    双侧腋下及锁骨上下区淋巴结[7. 淋巴结情况]。
    临床诊断：[8. 诊断结果]

    --------------------------------------------------
    【第三部分：诊断意见与后续建议】

    请结合影像表现和指南，给出总结性意见：

    1. **诊断意见**：
       - 基于AI的高置信度概率给出初步判断（如：高度疑似乳腺癌 / 疑似良性病变）。
       - **BI-RADS 分级评估**：结合指南，AI概率和病灶形态，给出可能的 BI-RADS 分级（如：4, 5 或 2, 3类）。
       - **指南依据**：请**直接引用**【参考指南】中的相关文本作为支撑，指南依据要分行写。
         *例如：“根据指南片段1，不规则形态且纵横比>1提示恶性...”*
         *(若参考指南中无直接对应文字，请引用通用的 BI-RADS 标准作为依据)*

    2. **治疗建议**：
       - **处理方案**：是否需要活检或随访（如：超声引导下穿刺、MRI增强、定期复查等）。
       - **指南流程**：引用指南中关于该类病变(如BI-RADS 4/5类)的标准处理流程。
    --------------------------------------------------
    """

    prompt = PromptTemplate.from_template(template_text)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": rag_context, "question": combined_report_text})

def combine_reports(filename, cls_text, det_text):
    return f"""
    === 文件名: {filename} ===
    [AI分类数据]
    {cls_text}
    
    [AI检测数据]
    {det_text}
    """

def run_analysis_process(cls_path, det_path):
    """
    主执行逻辑：接收路径 -> 读取文件 -> 调用LLM -> 输出结果
    """
    llm = ChatOpenAI(
        temperature=0.3, 
        model="deepseek-chat", 
        # model = "gpt-5",
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL
    )

    print("=" * 60)
    print("正在初始化分析...")
    
    # 读取文件
    txt_cls = load_file_content(cls_path)
    txt_det = load_file_content(det_path)

    if not txt_cls or not txt_det:
        print("❌ 无法获取完整的报告数据，程序终止。")
        return

    # 使用检测报告的文件名作为案例名
    file_name = os.path.basename(det_path)

    print("-" * 60)
    print(f"开始生成报告: {file_name}")
    print("-" * 60)
    
    # 执行分析
    result = ask_breast_analysis(llm, combine_reports(file_name, txt_cls, txt_det))
    print(result)

if __name__ == "__main__":
    # 配置命令行参数解析
    parser = argparse.ArgumentParser(description="基于AI分类和检测报告生成乳腺诊断报告")
    
    parser.add_argument(
        "--cls", 
        type=str, 
        default=DEFAULT_CLS_PATH,
        help="分类报告(.txt)的完整路径"
    )
    parser.add_argument(
        "--det", 
        type=str, 
        default=DEFAULT_DET_PATH,
        help="检测/分割报告(.txt)的完整路径"
    )

    args = parser.parse_args()

    # 运行主流程
    run_analysis_process(args.cls, args.det)