from pathlib import Path
from langchain_core.tools import tool
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_core.output_parsers import JsonOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

# 0. 基础配置
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOC_PATH = PROJECT_ROOT / 'data' / 'company_handbook.md'
VECTOR_DIR = PROJECT_ROOT / 'db' / 'chroma.db'

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
RERANKER_MODEL = os.getenv("RERANK_MODEL")

# 1. 初始化核心模型组件
print('正在加载 BGE 嵌入模型。。。。。。')
embeddings = HuggingFaceEmbeddings(
    model_name = os.getenv('EMBEDDING_MODEL'),
    model_kwargs = {'device': 'mps'},       # 如果是 英伟达显卡可以填写 cuda，苹果M芯片填写 mps， 其他填写 cpu 或这个参数都不写
    encode_kwargs = {'normalize_embeddings': True}
)

print('正在加载 BGE Reranker 模型。。。。。。')
reranker = CrossEncoder(RERANKER_MODEL, max_length=512, device='mps')

# 由于 llm 用于 Query改写 和 HyDE生成，温度设置为 0.7 可以激发一定的创造力
llm = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.7
)

# 2. 构建多路召回 Retriver
def build_ensemble_retriver():
    """构建 BM25 + Vector的混合检索器"""
    if not DOC_PATH.exists():
        raise FileNotFoundError(f'找不到知识库文件：{DOC_PATH}')

    with open(DOC_PATH, 'r', encoding='utf-8') as f:
        markdown_text = f.read()

    # 第一层：基于 Makdown 层级进行切分
    headers_to_split_on = [
        ('##', 'Chapter'),
        ('###', 'Section')
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_text)

    # 第二层：按照['\n\n','\n'] 顺序递归切分
    # 第三层：为了防止么某个章节依然过长，再叠加一个字符集滑动窗口切分兜底
    chunk_size = 500
    chunk_overlap = 50
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=['\n\n','\n']
    )
    splits = text_splitter.split_documents(md_header_splits)

    print(f'文档切分完毕，共生成 {len(splits)} 个语义文本块(chunks)。正在存入数据库')

    # 路线A： 全文关键字检索（BM25）
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 5

    # 路线B：向量语义检索
    if VECTOR_DIR.exists() and any(VECTOR_DIR.iterdir()):       # 已经存在向量数据，直接加载
        print('「知识库构建」检测到本地持久化向量库，直接加载。。。。。')
        vectorstore = Chroma(persist_directory=str(VECTOR_DIR),
                      embedding_function=embeddings)
    else:
        print('「知识库构建」本地无缓存，正在生成向量数据库并落盘。。。。。')
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=str(VECTOR_DIR),
        )
    voctor_retriever = vectorstore.as_retriever(search_kwargs={'k': 5})

    # 混合：使用 EnsembleRetriever （底层采用 事 RRF 倒数秩融合算法）
    ensemble_retriever =  EnsembleRetriever(
        retrievers=[bm25_retriever, voctor_retriever],
        weights=[0.4, 0.6]      # 权重可调：偏向关键字还是偏向语义， 40%依赖关键字，60%依赖语义泛化
    )
    return ensemble_retriever

print('正在构建混合检索器.......')
retriever = build_ensemble_retriver()   # 启动时初始化 build_ensemble_retriver

# 3. 智能扩写与 HyDE
class QueryExpansion(BaseModel):
    expanded_queries: list[str] = Field(description='从不同维度扩写 3 个相关检索词或短语')
    hypothetical_document:str = Field(description='针对该问题的一段假设性、看似专业的官方制度回答片段（允许伪造数字）')

expansion_parser = JsonOutputParser(pydantic_object=QueryExpansion)

def expand_and_hyde(original_query:str) -> list[str]:
    """利用 LLM 生成多维度扩写与 HyDE 假设"""
    prompt = ChatPromptTemplate.from_template(
        "你是一名专业的企业 HR 专家。为了提高提高知识库检索命中率，请协助处理用户的原始提问。\n"
        "任务 1（多维扩写）：站在不同视角（如政策名次、审批流程、系统操作）扩写 3 个相关检索词或短语。\n"
        "任务 2（HyDE假设）：用官方、严谨的 HR 规章制度口吻，伪造一段回答该问题的文本。不管事实是否正确，重点是极度模仿‘员工手册’"
        "的很专业行文风格和词汇分布。\n\n"
        "用户原始问题：{query}\n\n"
        "{format_instructions}"
    )

    chain = prompt | llm | expansion_parser

    try:
        result = chain.invoke({
            'query': original_query,
            'format_instructions': expansion_parser.get_format_instructions()
        })

        print(f'\n原始问题：‘{original_query}’')
        print(f'        -> 衍生查询：{result['expanded_queries']}')
        print(f'        -> HyDE伪文：{result['hypothetical_document'][:30]}.......')

        # 汇总：原始问题 + 3 个衍生问题 + 1 个假设文档
        return [original_query] + result['expanded_queries'] + [result['hypothetical_document']]

    except Exception as e:
        print(f' LLM 调用失败， 降级使用基础检索。原因：{e}')
        return [original_query]

# 4. 封装成工具
@tool
def search_hr_policy(query:str) -> str:
    """
    高级知识搜索引擎（具备自动改写、混合检索、重拍功能）。
    当用户询问任何关于公司规章制度、差旅报销标准、假期政策、福利等相关信息，必须调用此工具。
    输入参数 query 必须是用户原始问题
    """
    # 步骤一：获取 5 个查询变体组成的查询矩阵
    search_queries = expand_and_hyde(query)

    # 步骤二：多路并发检索（BM25 + Vector）
    all_condition_docs = []
    for q in search_queries:
        docs = retriever.invoke(q)
        all_condition_docs.extend(docs)

    # 步骤三：文档去重（以文档内容作为唯一标识）
    unique_docs = {doc.page_content: doc for doc in all_condition_docs}.values()
    unique_docs = list(unique_docs)

    if not unique_docs:
        return '知识库中未检索到相关政策，请提示用户询问 HR 人工。'

    # 步骤四： Cross-Encoder  （交叉编码器）精准重排
    # 必须用用户「原始真实问题」去和召回的文档计算相关性得分
    sentence_pairs = [[query, doc.page_content] for doc in unique_docs]
    scores = reranker.predict(sentence_pairs)

    scored_doc = list(zip(unique_docs, scores))
    # 按模型打分从高到低排序
    scored_doc.sort(key=lambda x: x[1], reverse=True)

    # 步骤五：截取真正的 Top-3 并组装返回文本
    top_3_docs = [doc for doc, _ in scored_doc[:3]]

    context_parts = []
    for i, doc in enumerate(top_3_docs, 1):
        chapter = doc.metadata.get('Chapter', '未知章节')
        section = doc.metadata.get('Section', '未知段落')
        context_parts.append(f'「来源 {i}」 {chapter} > {section} \n {doc.page_content}')

    merged_context = '\n\n'.join(context_parts)

    return f'「知识库检索结果」\n{merged_context}'
