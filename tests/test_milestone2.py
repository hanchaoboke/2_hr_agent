import sys
from pathlib import Path

# 为了防止 ModuleNotFoundError ，我们加入下面两行代码，可以自动将项目根目录挂载到系统环境变量中，提高鲁棒性
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from agent.rag_pipeline import search_hr_policy

QUESTIONS = [
    'P5员工去成都出差，一天住宿报销多少钱？',
    '入职半年的新人公司有福利假期么？',
    '我想开收入证明，可以在系统里弄么？'
]

import pytest

@pytest.mark.parametrize('question', QUESTIONS)
def test_search_hr_policy(question):
    result = search_hr_policy.invoke({'query': question})

    assert isinstance(result, str)
    assert result.strip() , '检索结果不应为空'
    assert '来源' in result, f'未召回任何知识库来源，实际返回 {result}'
    assert '未检索到相关政策' not in result, '不应落到“未检测到”的兜底分支'



if __name__ == '__main__':
    for i, question in enumerate(QUESTIONS, 1):
        print(f'\n 「测试询问 {i}」: {question}')
        print('-'*50)

        result = search_hr_policy.invoke({'query':question})
        print(result)
        print('-' * 50)