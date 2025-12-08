import asyncio
import traceback
from langchain_core.messages import (
    AnyMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.func import task, entrypoint
from langchain_core.language_models import LanguageModelLike
from components import make_chroma, llm, llm2
from config import *
import json
from mem0.memory.utils import remove_code_blocks, extract_json
from mem0.configs.prompts import get_update_memory_messages
from utils import get_date

logger = get_log(__name__)

# KEEP_MSGS = 4
# MSGS_THRESH = 10
KEEP_MSGS = 20
MSGS_THRESH = 50

# REF: mem0.configs.prompts.AGENT_MEMORY_EXTRACTION_PROMPT, assisted by Prof. DeepSeek
MEMORY_EXTRACTION_PROMPT = """
接下来，请执行一个专门的信息整理任务。严格遵循以下指令：

你是信息整理器，专门负责从你之前的交互历史上下文中，准确提取出外界环境相关的事实。你需要从上下文中提取信息，并将其整理为清晰、易于管理的事实条目，这些信息将保存在记忆系统中，便于在未来的交互中检索和回忆。以下是需要重点关注的信息类型以及如何处理输入数据的详细说明。

注意：不要从系统消息中提取信息。用户消息中“相关记忆”部分是你已经提取并记住的信息片段，无需重复提取。

需要记忆的信息类型：

- **人物相关**：记录人物在各种类别（如活动、感兴趣的话题、假设场景）中提到的喜好、厌恶及具体偏好，注意人物提及的任何特定技能、知识领域或能够执行的任务，注意人物提及的任何特定技能、知识领域，记录人物描述的假设性活动或计划。
- **人物偏好**：记录人物在各种类别（如活动、感兴趣的话题、假设场景）中提到的喜好、厌恶及具体偏好。
- **人物能力**：注意人物提及的任何特定技能、知识领域或能够执行的任务。
- **人物的假设计划或活动**：记录人物描述的假设性活动或计划。
- **环境状态**：记录你通过工具与外界交互时。
- **其他信息**：记录人物分享的关于自身的任何其他有趣或独特的细节。

以下是一些少量示例：

约翰：你好，我在旧金山找一家餐厅。
助手：好的，我能帮忙。您有什么特别想吃的菜系吗？
输出：{{"facts": ["约翰想在旧金山找一家餐厅"]}}

约翰：昨天下午3点我和亚历克斯开了个会，我们讨论了新项目。
助手：听起来是个高效的会议。
输出：{{"facts": ["昨天下午3点约翰和亚历克斯开会讨论了新项目"]}}

约翰：你好，我叫约翰，是一名软件工程师。
助手：很高兴认识你，约翰！有什么可以帮你的？
输出：{{"facts": ["约翰是软件工程师"]}}

亚历克斯：我最喜欢的电影是《盗梦空间》和《星际穿越》。
助手：好选择！这两部都是很棒的电影。
输出：{{"facts": ["亚历克斯最喜欢的电影是《黑暗骑士》和《肖申克的救赎》"]}}

请按照以上示例的JSON格式返回事实和偏好。

请记住：
- 当前时间是{date}。
- 不要返回上方提供的自定义少量示例提示中的任何内容。
- 如果没有提取到任何事实，返回与"facts"键对应的空列表。
- 请确保按照示例中提到的格式返回响应。响应应为JSON格式，键为"facts"，对应的值为字符串列表。
- 使用最能保持信息原样的语言记录事实。

现在，请开始执行此任务。直接按照上述JSON格式返回。
"""

MEMORY_QUERY_PROMPT = """
接下来，你要执行的任务是：

根据交互上下文的当前状态，提取查询语句。这些查询语句将用于从基于向量数据库的记忆系统中查询相关记忆信息，以支持你的后续行为。
"""


async def llm_xjson(model: LanguageModelLike, msg):
    for retry in range(10):
        logger.info("llm_xjson invoking llm")
        try:
            ret = await model.ainvoke(msg)
        except Exception:
            logger.error(f"llm_xjson llm invoke error: {traceback.format_exc()}")
            await asyncio.sleep(retry + 3)
        text = ret.text  # type: ignore
        logger.info(f"llm_xjson llm returned: {text}")
        try:
            result = json.loads(extract_json(remove_code_blocks(text)))
            return result
        except json.JSONDecodeError as e:
            logger.warning(
                f"llm_xjson JSONDecodeError: {e}: llm returned invalid json: {text}"
            )
    raise Exception("llm_xjson: too many retries")


chroma = make_chroma("memorying", DATADIR + "/chroma")


async def extract_facts(model: LanguageModelLike, msgs: list[AnyMessage]) -> list[str]:
    try:
        result = await llm_xjson(
            model,
            msgs
            + [
                SystemMessage(MEMORY_EXTRACTION_PROMPT.format(date=get_date())),
                HumanMessage("直接输出结果："),
            ],
        )
        return list(map(str.strip, result.get("facts", [])))
    except Exception as e:
        logger.error(f"fact extraction failed: {e}")
    return []


async def update_memory(facts: list[str], results: list):
    if len(results) == 0:
        await chroma.aadd_texts(list(facts))
        return
    prompt = get_update_memory_messages(
        [{"id": str(i[0]), "text": i[1].page_content} for i in enumerate(results)],
        list(facts),
    )
    logger.info("sending memory update query")
    actions = (await llm_xjson(llm2, prompt)).get("memory", [])
    adds = []
    update_ids = []
    updates = []
    deletes = []
    for action in actions:
        text = action.get("text").strip()
        if not text:
            continue
        event = action.get("event")
        idx = action.get("id")
        if event == "ADD":
            adds.append(text)
            continue
        if isinstance(idx, str) and idx.isdigit():
            idx = int(idx)
        elif not isinstance(idx, int):
            continue
        if event == "UPDATE":
            update_ids.append(results[idx].id)
            updates.append(text)
        elif event == "DELETE":
            deletes.append(results[idx].id)
    logger.info(f"adds: {adds} updates: {updates} deletes: {deletes}")
    deletes += update_ids
    if len(deletes) > 0:
        await chroma.adelete(deletes)
    if len(adds) > 0:
        await chroma.aadd_texts(adds)
    if len(updates) > 0:
        await chroma.aadd_texts(updates, ids=update_ids)
    return


@task
async def process_memory(model: LanguageModelLike, msgs: list[AnyMessage]):
    logger.info("process_memory start")
    facts = await extract_facts(model, msgs)
    logger.info(f"extracted facts: {facts}")
    ths = [chroma.asearch(fact, "mmr") for fact in facts]
    search_results = await asyncio.gather(*ths)
    unique_results = list(
        {doc.id: doc for doc_ in search_results for doc in doc_}.values()
    )
    logger.info(f"searched existing memories: {unique_results}")
    await update_memory(facts, list(unique_results))
    logger.info("process_memory finish")


def main():
    @entrypoint()
    async def test_main(_):
        await process_memory(
            llm,
            [
                SystemMessage("现在时间：2023年3月2日"),
                HumanMessage("我家在杭州，你家在哪？"),
                AIMessage("我家在长春。"),
            ],
        )
        await process_memory(
            llm,
            [
                SystemMessage("现在时间：2023年12月2日"),
                HumanMessage("我家在杭州，你家在哪？"),
                AIMessage("我最近搬家了，现在家在上海。"),
                HumanMessage("你现在在做什么工作？"),
                AIMessage("我还在上学，刚上大三。"),
                HumanMessage("你家离我挺近的，什么时候见个面？"),
                AIMessage("好啊，要不就下个月吧？正好我元旦想去杭州玩"),
                HumanMessage("好，我等你来！"),
            ],
        )
        await process_memory(
            llm,
            [
                SystemMessage("现在时间：2024年1月23日"),
                HumanMessage("上次来杭州吃的西湖醋鱼，你觉得怎么样？"),
                AIMessage("感觉味道怪怪的，我不是很喜欢"),
            ],
        )

    asyncio.run(test_main.ainvoke({}))


if __name__ == "__main__":
    main()
