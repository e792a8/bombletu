from pydantic_ai import ModelMessage, ModelResponse, ToolDefinition, ToolReturn
from components import langfuse
from pydantic_ai.models import Model
from pydantic_ai.direct import model_request
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.messages import ModelRequest, SystemPromptPart, UserPromptPart


KEEP_ROUNDS = 4


async def summarize(
    model: Model, tool_defs: list[ToolDefinition], msgs: list[ModelMessage]
):
    reduce_prompt = langfuse.get_prompt("context-reduce").prompt
    cut = 0
    tail_rounds = 0
    for i in range(len(msgs) - 2, -1, -1):
        if (
            isinstance(msgs[i], ModelResponse)
            or isinstance(msgs[i].parts[0], ToolReturn)
        ) and isinstance(msgs[i + 1], ModelRequest):
            tail_rounds += 1
            if tail_rounds >= KEEP_ROUNDS:
                cut = i + 1
                break
    if cut <= 2:
        return None, None
    msgs_to_sum = msgs[:cut]
    msgs_to_keep = msgs[cut:]

    resp = await model_request(
        model,
        msgs_to_sum
        + [
            ModelRequest(
                parts=[
                    SystemPromptPart(reduce_prompt),
                    UserPromptPart("输出提取的交互上下文摘要:"),
                ]
            )
        ],
        model_request_parameters=ModelRequestParameters(
            function_tools=tool_defs, output_mode="text"
        ),
    )

    return resp.text, msgs_to_keep
