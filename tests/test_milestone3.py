import sys
import io
from PIL import Image as PILImage
from pathlib import Path
from langchain_core.messages import HumanMessage

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from agent.graph_builder import hr_agent_app

def display_graph(graph, xray_deep=1):
    try:
        png_data = graph.get_graph(xray=xray_deep).draw_mermaid_png()
        img = PILImage.open(io.BytesIO(png_data))
        img.show()
        img.save("my_graph.png")
        print(' 架构图已成功弹窗！')
    except Exception as e:
        print(f'出错：{e}')


def chat_with_agent(uid:str, question:str):
    """与HR Agent 进行流式交互测试"""
    print('==============================================')
    print(f'「员工 UID：{uid}」 提问：{question}')
    print('==============================================')

    initial_state = {
        'messages':[HumanMessage(content=question)],
        'current_uid':uid,
        'loop_step':0,
    }

    # 启用流式输出，方便观察内部执行路径
    for event in hr_agent_app.stream(initial_state, stream_mode='values'):
        last_msg = event['messages'][-1]

        # 过滤掉系统初始输入和反思审计的打回提示，让展示日志干净一些
        if isinstance(last_msg, HumanMessage) and '[SYSTEM AUDIT FAILED]' not in last_msg.content:
            continue

        if last_msg.type == 'ai' and not last_msg.tool_calls:
            print(f'\n 「AI 最终答复」：\n{last_msg.content}\n')
        if last_msg.type == 'ai' and last_msg.tool_calls:
            for tool in last_msg.tool_calls:
                print(f'    「调度工具」 -> {tool['name']} ({tool['args']})')


if __name__ == '__main__':
    # display_graph(hr_agent_app)
    # 简单数据库操作
    print('============ 简单数据库操作 ============')
    chat_with_agent(uid='1002', question='帮我查一下我还有几天年假？如果可以的话顺便帮我开一个证明')
    print('============ 简单数据库操作 结束 ============')

    # 触发 RAG
    print('============ 测试2 ============')
    chat_with_agent(uid='1002', question='我下周要去北京出差，住宿费最高报销多少？')
    print('============ 测试2 结束 ============')

    # 触发 RAG
    print('============ 测试3 ============')
    chat_with_agent(uid='1001', question='我刚入职两年，如果休3天事假，需要谁来审批？')
    print('============ 测试3 结束 ============')

