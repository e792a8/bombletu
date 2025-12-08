from ncatbot.core import BotClient
from ncatbot.core.event import GroupMessageEvent
from oicq.applet import OicqApplet
import asyncio
import os
import signal
import traceback
from config import *
from agent import Agent

logger = get_log(__name__)


qbot = BotClient()


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
    await asyncio.sleep(1)
    await qbot.api.send_group_text(
        GRP,
        "Someone tell [CQ:at,qq=1571224208] there is a problem with my AI.",
    )
    await asyncio.sleep(1)
    await qbot.api.send_group_text(
        CON, f"{GRP} {traceback.format_exception(e)} delay: {retry_delay}"
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
    return mcp_config


def main():
    q = OicqApplet()
    agent = Agent([q], get_mcp_config())
    asyncio.run(agent.run(alarm))


if __name__ == "__main__":
    main()
