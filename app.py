"""飞羽科技 HR 智能助理 - Streamlit 前端

直接 import hr_agent_app 进程内调用，不依赖 api.py。
运行：streamlit run app.py
"""

import uuid

import streamlit as st
from langchain_core.messages import HumanMessage

TOOL_LABELS = {
    'get_employee_profile': '查询员工档案',
    'get_leave_balance': '查询假期余额',
    'generate_employment_certificate': '生成证明文件',
    'search_hr_policy': '检索公司政策知识库',
}


@st.cache_resource(show_spinner='正在加载 HR Agent（含本地模型，首次较慢）...')
def load_agent():
    """缓存 LangGraph 应用，避免每次 rerun 重复加载 BGE 模型"""
    from agent.graph_builder import hr_agent_app
    return hr_agent_app


def init_session_state():
    if 'thread_id' not in st.session_state:
        st.session_state.thread_id = uuid.uuid4().hex[:8]
    if 'messages' not in st.session_state:
        st.session_state.messages = []  # [{'role': 'user'/'assistant', 'content': str}]


def render_sidebar():
    with st.sidebar:
        st.title('飞羽科技 HR 助理')
        uid = st.text_input('员工 UID', value=st.session_state.get('uid', '1001'))
        st.session_state.uid = uid.strip()

        st.caption(f'当前会话 ID：`{st.session_state.thread_id}`')
        if st.button('新建会话', use_container_width=True):
            st.session_state.thread_id = uuid.uuid4().hex[:8]
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption('可咨询：考勤/报销/福利政策、年假余额、在职/收入证明开具等。')


def run_agent(question: str):
    """调用 Agent，返回 (工具调用序列, 答复文本生成器)"""
    agent = load_agent()
    config = {'configurable': {'thread_id': st.session_state.thread_id}}
    state = {
        'messages': [HumanMessage(content=question)],
        'current_uid': st.session_state.uid,
        'loop_state': 0,
    }
    return agent.stream(state, config, stream_mode='values')


def answer_question(question: str):
    """驱动图执行，展示工具进度并流式输出最终答复"""
    final_text = ''

    with st.status('Agent 思考中...', expanded=True) as status:
        placeholder = st.empty()
        try:
            for event in run_agent(question):
                last_msg = event['messages'][-1]

                if isinstance(last_msg, HumanMessage):
                    continue

                if last_msg.type == 'ai' and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        label = TOOL_LABELS.get(tc.get('name'), tc.get('name', 'unknown'))
                        status.write(f'🔧 正在{label}...')
                    continue

                if last_msg.type == 'tool':
                    label = TOOL_LABELS.get(getattr(last_msg, 'name', ''), '工具')
                    status.write(f'✅ {label}完成')
                    continue

                if last_msg.type == 'ai' and last_msg.content:
                    final_text = last_msg.content
                    placeholder.markdown(final_text + '▌')

            status.update(label='回答完毕', state='complete', expanded=False)
        except Exception as e:
            status.update(label='服务异常', state='error', expanded=True)
            st.error(f'Agent 调用失败：{type(e).__name__}: {e}')
            return None

    if not final_text:
        st.warning('Agent 未生成答复，请稍后重试。')
        return None

    placeholder.markdown(final_text)
    return final_text


def main():
    st.set_page_config(page_title='飞羽科技 HR 助理', page_icon='💼')
    init_session_state()
    render_sidebar()

    st.header('💼 智能 HR / 行政助理')

    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    question = st.chat_input('请输入您的问题，例如：我还有多少天年假？')
    if not question:
        return
    if not st.session_state.uid:
        st.warning('请先在左侧边栏填写员工 UID。')
        return

    with st.chat_message('user'):
        st.markdown(question)
    st.session_state.messages.append({'role': 'user', 'content': question})

    with st.chat_message('assistant'):
        answer = answer_question(question)
    if answer:
        st.session_state.messages.append({'role': 'assistant', 'content': answer})


if __name__ == '__main__':
    main()
