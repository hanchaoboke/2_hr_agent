"""HR 智能助理 FastAPI 接口层

提供：
- POST /chat   : SSE 流式聊天接口（工具进度 + 最终答复）
- GET  /health : 健康检查

SSE 帧协议（每帧为 `data: <json>\n\n`）：
- {"type": "tool_call", "tools": [...]}        Agent 正在调用工具
- {"type": "tool_result", "tool": ..., "preview": ...}  工具返回摘要
- {"type": "answer", "content": ...}           最终答复（完整文本）
- {"type": "error", "message": ...}            异常
- {"type": "done"}                             本次问答结束（最后一个帧）

注意：InMemorySaver 记忆与 SQLite 连接均为进程内状态，必须以单 worker 运行 uvicorn。
"""

import json
from typing import Generator

from typing import Any, Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from agent.graph_builder import hr_agent_app

app = FastAPI(title='HR Agent API', version='1.0.0')

# 允许前端跨域调用（前端通常以 file:// 或独立静态服务承载，与 8000 端口不同源）
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'],
)


class ChatRequest(BaseModel):
    uid: str = Field(..., min_length=1, description='员工 UID，如 1001')
    thread_id: str = Field(..., min_length=1, description='会话 ID，相同 ID 共享记忆')
    question: str = Field(..., min_length=1, description='用户提问')

class ApiChatRequest(BaseModel):
    uid: str = Field(..., min_length=1, description='员工 UID，如 1001')
    thread_id: str = Field(..., min_length=1, description='会话 ID，相同 ID 共享记忆')
    message: str = Field(..., min_length=1, description='用户提问')

class ResumeRequest(BaseModel):
    """指向之前已经出发 interrupt()的会话"""
    thread_id: str = Field(..., min_length=1)
    action: Literal['approve', 'reject']

class ChatResponse(BaseModel):
    """JSON 内的业务码，不是 HTTP 状态码
        200=图完成；202 图挂起，等待人类审批"""
    code: int = 200
    thread_id: str
    # completed 表示流程已经结束；waiting_for_review表示仍然需要审批
    status: Literal['completed', 'waiting_for_review']
    answer: str = ''
    interrupt_msg: str = ''


# 读取 LangGraph 的中断提示
def _interrupt_message(snapshot: Any)->str:
    """从 LangGraph StateSnapshot 中取出 interrupt 文案"""
    interrupts = getattr(snapshot, 'interrupts', None) or ()
    # 兼容部分老版本
    if not interrupts:
        for task in getattr(snapshot, 'tasks', ()) or ():
            interrupts = getattr(task, 'interrupts', None) or ()
            if interrupts:
                break
    if not interrupts:
        # 如果暂时看不到具体文案，使用默认提示
        return ' Agent 正在尝试执行敏感操作，等待人工授权。'
    return str(getattr(interrupts[0], 'value', interrupts[0]))

# 统一运行 LangGraph
def _run_graph(input_value:Any, config: dict[str, Any]) ->tuple[str, str]:
    """
    运行图并回答（最终回答，中断提示）
    input_value：新问题时传入 state 字典
    config：审批恢复时，传入 Command(resume=action)
    """
    snapshot = hr_agent_app.get_state(config)
    prior_messages = (
        snapshot.values.get('messages', [])
        if snapshot and snapshot.values else []
    )
    cursor = len(prior_messages)
    final_answer = ''

    for event in hr_agent_app.stream(input_value, config, stream_mode='values'):
        messages = event.get('messages', [])
        for message in messages[cursor:]:
            if isinstance(message, HumanMessage):
                continue
            if message.type == 'ai' and not message.tool_calls and message.content:
                final_answer = str(message.content)
        cursor = len(messages)
    # stream 结束有两种情况：正常到达 END 或者 interrupt() 挂起
    snapshot = hr_agent_app.get_state(config)
    if snapshot and snapshot.next:
        return '', _interrupt_message(snapshot)
    return  final_answer, ''

# 把图运行结果转换为统一响应
def _build_response(thread_id:str, final_answer:str, interrupt_msg:str) -> ChatResponse:
    if interrupt_msg:
        return ChatResponse(
            code=202,
            thread_id=thread_id,
            status='waiting_for_review',
            interrupt_msg=interrupt_msg,
        )
    if not final_answer:
        # 即没有挂起又没有回答，说明图执行结果不符预期
        raise HTTPException(
            status_code=500,
            detail='Agent 未生成有效回答'
        )
    return ChatResponse(
        code=200,
        thread_id=thread_id,
        status='completed',
        answer=final_answer,
    )

def _sse_frame(payload: dict) -> str:
    return f'data: {json.dumps(payload, ensure_ascii=False)}\n\n'


def _chat_event_stream(req: ChatRequest) -> Generator[str, None, None]:
    """驱动 LangGraph 并将中间事件转换为 SSE 帧。

    使用「消息游标」处理流中新增的消息（而非只看最后一条），以保证：
    - 并行调用多个工具时，每个 tool_result 都能上报；
    - fact_checker 返回空消息时不重复上报同一 answer 帧。
    """
    config = {'configurable': {'thread_id': req.thread_id}}
    state = {
        'messages': [HumanMessage(content=req.question)],
        'current_uid': req.uid,
        'loop_state': 0,
    }

    # 基线：该线程已持久化的历史消息数，用于跳过旧消息、只处理本轮新增（多轮记忆共享）
    try:
        snapshot = hr_agent_app.get_state(config)
        prior = snapshot.values.get('messages', []) if snapshot and snapshot.values else []
        cursor = len(prior)
    except Exception:
        cursor = 0

    answered = False
    try:
        for event in hr_agent_app.stream(state, config, stream_mode='values'):
            messages = event['messages']

            for msg in messages[cursor:]:
                if isinstance(msg, HumanMessage):
                    continue

                # Agent 决定调用工具：上报进度
                if msg.type == 'ai' and msg.tool_calls:
                    tool_names = [tc.get('name', 'unknown') for tc in msg.tool_calls]
                    yield _sse_frame({'type': 'tool_call', 'tools': tool_names})
                    continue

                # 工具执行结果：上报摘要
                if msg.type == 'tool':
                    yield _sse_frame({
                        'type': 'tool_result',
                        'tool': getattr(msg, 'name', '') or 'unknown',
                        'preview': str(msg.content)[:100],
                    })
                    continue

                # 最终答复（含审计打回后的重写结果，只发非空内容）
                if msg.type == 'ai' and msg.content:
                    yield _sse_frame({'type': 'answer', 'content': msg.content})
                    answered = True

            cursor = len(messages)

        if not answered:
            yield _sse_frame({'type': 'error', 'message': 'Agent 未生成答复，请稍后重试'})
    except Exception as e:
        print(f'「API异常」图执行失败：{type(e).__name__}: {e}')
        yield _sse_frame({'type': 'error', 'message': f'服务内部错误：{type(e).__name__}'})
    finally:
        yield _sse_frame({'type': 'done'})


@app.post('/chat')
def chat(req: ChatRequest):
    """SSE 流式聊天接口。同步 def 端点由 Starlette 线程池执行，避免阻塞事件循环"""
    return StreamingResponse(
        _chat_event_stream(req),
        media_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )

# 发起对话接口
@app.post('/api/v1/chat', response_model=ChatResponse)
def api_chat(request: ApiChatRequest):
    config = {'configurable': {'thread_id': request.thread_id}}
    try:
        snapshot = hr_agent_app.get_state(config)
        if snapshot and snapshot.next:
            # 检查这个 thread_id 是否已经在等待人工审批
            return  ChatResponse(
                code=202,
                thread_id=request.thread_id,
                status='waiting_for_review',
                interrupt_msg=_interrupt_message(snapshot),
            )
        state = {
            'messages':[HumanMessage(content=request.message)],
            'current_uid': request.uid,
            'loop_state': 0,
        }
        final_answer, interrupt_msg = _run_graph(state, config)
        return _build_response(
            request.thread_id,
            final_answer,
            interrupt_msg,
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f'API Chat error: {type(e).__name__}: {e}')
        raise HTTPException(status_code=500, detail=f'内部推理引擎错误：{type(e).__name__}') from e

# 人工恢复接口
@app.post('/api/v1/resume', response_model=ChatResponse)
def api_resume(request: ResumeRequest):
    config = {'configurable': {'thread_id': request.thread_id}}
    snapshot = hr_agent_app.get_state(config)

    # next 为空说明该会话没有待恢复节点，不能调用 resume
    if not snapshot or not snapshot.next:
        raise HTTPException(
            status_code=400,
            detail='当前会话未处于等待审批状态'
        )
    try:
        final_answer, interrupt_msg = _run_graph(Command(resume=request.action),config)
        return _build_response(
            request.thread_id,
            final_answer,
            interrupt_msg,
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f'API Resume error: {type(e).__name__}: {e}')
        raise HTTPException(status_code=500, detail=f'内部推理引擎错误：{type(e).__name__}') from e


@app.get('/health')
def health():
    return {'status': 'ok', 'service': 'hr-agent'}


if __name__ == '__main__':
    import uvicorn
    # 必须单 worker：InMemorySaver 记忆为进程内状态
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
