import json
from langchain.messages import (
    SystemMessage,
    HumanMessage,
)
from pydantic_ai import ModelMessage, SystemPromptPart, UserPromptPart, ModelRequest
from agenting.types import BotDeps, BotState
from config import *
from components import langfuse


def initial_prompts(deps: BotDeps, state: BotState) -> list[ModelMessage]:
    system_prompt = langfuse.get_prompt("system-prompt").prompt

    try:
        with open(DATADIR + "/note.json", "r") as f:
            note = json.load(f)
        if not isinstance(note, list):
            note = []
    except (json.JSONDecodeError, FileNotFoundError):
        note = []
    notes = [f"{i + 1} {n}" for i, n in enumerate(note)]
    if len(notes) == 0:
        notes = "当前无笔记"
    else:
        notes = "\n".join(notes)

    return [
        ModelRequest(
            [
                SystemPromptPart(
                    "\n".join(
                        [
                            system_prompt,
                            deps.applet_instructions,
                            f"当前你的笔记内容(按需使用`edit_note`编辑笔记):\n```\n{notes}\n```\n",
                            f"之前的交互历史摘要:\n```\n{state.summary}\n```\n",
                        ]
                    )
                ),
                UserPromptPart("输出你的逐步推理思考过程，并调用工具执行动作:"),
            ]
        )
    ]
