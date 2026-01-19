import asyncio
import os
import signal
import traceback
from pathlib import Path

from ncatbot.core.event import GroupMessageEvent
from pydantic_ai.mcp import MCPServerConfig
from pydantic_ai.toolsets import CombinedToolset
from pydantic_graph import End
from pydantic_graph.persistence.file import FileStatePersistence

from agenting.graph import Idle, graph
from agenting.tools import local_toolset
from agenting.types import BotDeps, BotState
from config import *
from oicq.globl import qbot
from oicq.status import init_read_status
from oicq.tools import oicq_toolset

logger = get_log(__name__)


@qbot.on_group_message()  # type: ignore
async def group_message_handler(event: GroupMessageEvent):
    if event.group_id == CON:
        if event.raw_message == "/kill":
            logger.warning("kill called")
            os.kill(os.getpid(), signal.SIGKILL)
        elif event.raw_message == "/panic":
            logger.info("panic called")
            os.kill(os.getpid(), signal.SIGTERM)


async def alarm(e: BaseException, retry_delay: float):
    # await asyncio.sleep(1)
    # await qbot.api.send_group_text(
    #     GRP,
    #     "Someone tell [CQ:at,qq=1571224208] there is a problem with my AI.",
    # )
    await asyncio.sleep(1)
    await qbot.api.send_group_text(
        CON,
        f"[CQ:at,qq=1571224208] {GRP} {traceback.format_exception(e)} delay: {retry_delay}",
    )


def get_mcp_config():
    mcp_config = {}
    for i in range(1, 100):
        if name := os.environ.get(f"MCP{i}_NAME"):
            mcp_config[name] = {
                "url": os.environ.get(f"MCP{i}_URL"),
                "transport": os.environ.get(f"MCP{i}_TRANSPORT"),
            }
        else:
            break
    return {"mcpServers": mcp_config}


async def amain():
    qbot.run_backend()

    config = MCPServerConfig.model_validate(get_mcp_config())

    servers = []
    for name, server in config.mcp_servers.items():
        server.id = name
        server.tool_prefix = name
        servers.append(server)

    await init_read_status()

    persist = FileStatePersistence(Path(DATADIR) / "graph_state.json")
    persist.set_graph_types(graph)
    if snapshot := await persist.load_next():
        state = snapshot.state
        node = snapshot.node
    else:
        state = BotState()
        node = Idle(None)
    deps = BotDeps(
        CombinedToolset[BotDeps]([oicq_toolset, local_toolset] + servers), ""
    )

    async with graph.iter(node, state=state, deps=deps, persistence=persist) as run:
        while True:
            node = await run.next()
            if isinstance(node, End):
                break
            logger.info(node)


def main():
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        pass
    finally:
        qbot.bot_exit()


if __name__ == "__main__":
    main()
