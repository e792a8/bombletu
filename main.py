from typing import List
from ncatbot.core import BotClient, MessageArray
from ncatbot.core.event import GroupMessageEvent
from asyncio_channel import create_channel, create_sliding_buffer
from graph import make_agent, config as agentconfig
import asyncio
from datetime import datetime
from pytz import timezone
from ncatbot.core.api import NapCatAPIError
import os, signal
from langchain.messages import AIMessage, ToolMessage, HumanMessage
from time import time
from msgfmt import msglfmt, parse_msg
from config import *

logger = get_log(__name__)


class App:
    async def wait_intr(self, minutes: int):  # FIXME
        target_time = time() + minutes * 60
        while True:
            intr = await self.intrchan.take(timeout=5)
            if intr is not None:
                return intr
            if time() > target_time:
                return None

    async def send(self, content: str, target=GRP):
        logger.info(f"send: {content}")
        try:
            await self.qbot.api.send_group_msg(target, parse_msg(content).to_list())
            return "[success]"
        except NapCatAPIError as e:
            logger.warning(f"get_message error: {e}")
            return "[error 软件暂时故障]"

    async def get_messages(self, fro: int, to: int, with_id: bool = False) -> str:
        logger.info(f"get_messages: {fro}, {to}, {with_id}")
        try:
            return msglfmt(
                (await self.qbot.api.get_group_msg_history(GRP, 0, fro))[
                    : fro - to + 1
                ],
                with_id,
            )
        except NapCatAPIError as e:
            logger.warning(f"get_message error: {e}")
            return "[error 软件暂时故障]"

    async def get_messages_by_id(self, id: str, before: int, after: int) -> str:
        api = self.qbot.api
        try:
            bef = await api.get_group_msg_history(GRP, id, before + 1, True)
            aft = await api.get_group_msg_history(GRP, id, after + 1, False)
            lst = bef + aft[1:]
            return msglfmt(lst, id)
        except NapCatAPIError as e:
            logger.error(f"{e}")
            return "[error 软件暂时故障]"

    async def collect_unread(self) -> int:
        collected = 0
        while ev := await self.newmsgchan.take(timeout=0):
            self.unread.append(ev)
            collected += 1
        return collected

    async def get_unread(self, limit: int) -> str:
        await self.collect_unread()
        l = []
        for _ in range(limit):
            if len(self.unread) < 1:
                break
            l.append(self.unread.pop(0))
        return "\n".join([msglfmt(l), f"[unread {len(self.unread)}]"])

    def __init__(self):
        global agentconfig  # FIXME when #6318 is ok
        agentconfig["configurable"]["app"] = self  # type: ignore

        async def group_message_handler(event: GroupMessageEvent):
            if event.group_id == GRP:
                await self.newmsgchan.put(event)
                if event.message.is_user_at(USR):
                    logger.info(
                        "group_message_handler: 提及我的消息 {event.raw_message} 送入intrchan"
                    )
                    await self.intrchan.put("提及我的消息")
            elif event.group_id == CON:
                if event.raw_message == "/kill":
                    logger.warning("panic called")
                    os.kill(os.getpid(), signal.SIGKILL)
                elif event.raw_message == "/term":
                    logger.info("exit called")
                    os.kill(os.getpid(), signal.SIGTERM)
                elif event.raw_message == "/int":
                    logger.info("exit called")
                    os.kill(os.getpid(), signal.SIGINT)

        qbot = BotClient()
        qbot.add_group_message_handler(group_message_handler)  # type: ignore
        qbot.add_startup_handler(make_agent_loop(self))  # type: ignore
        msgchan = create_channel(create_sliding_buffer(100))  # type: ignore
        intrchan = create_channel(create_sliding_buffer(1))  # type: ignore

        self.qbot = qbot
        self.newmsgchan = msgchan
        self.intrchan = intrchan
        self.unread = []

    def run(self):
        self.qbot.run_frontend()


def check_idle_call(ivk):
    if (
        "messages" in ivk
        and len(ivk["messages"]) > 0
        and isinstance(last_msg := ivk["messages"][-1], AIMessage)
        and len(last_msg.tool_calls) == 1
        and (call := last_msg.tool_calls[0])["name"] == "idle"
    ):
        call_id = call["id"]
        mins = int(call["args"]["minutes"])
        return call_id, mins
    return None, 0


async def agent_loop(app: App):
    logger.info("agent loop starting")
    agent = make_agent()
    msg_inject = []
    while True:
        logger.info("agent invoking")
        ret = await agent.ainvoke({"messages": msg_inject}, config=agentconfig, context=app, print_mode="updates")  # type: ignore
        logger.debug(f"agent return: {ret}")
        idle_id, mins = check_idle_call(ret)
        if idle_id:
            logger.info(f"agent sleeping {mins}min")
        else:
            logger.info(f"agent continuing")
        intr = await app.wait_intr(mins)
        unread = await app.collect_unread()
        msg_inject = []
        if idle_id is not None:
            if intr:
                msg_inject.append(
                    ToolMessage("Idle interrupted!", tool_call_id=idle_id)
                )
            else:
                msg_inject.append(ToolMessage("Idle finished.", tool_call_id=idle_id))
        info_inject = []
        if intr:
            info_inject.append(f"[notify {intr}]")
        if unread > 0:
            info_inject.append(f"[event {unread}条新消息]")
        if len(info_inject) > 0:
            msg_inject.append(HumanMessage("\n".join(info_inject)))


def make_agent_loop(app: App):
    async def agent_loop_wrapper(_):
        while True:
            await asyncio.sleep(10)
            try:
                await agent_loop(app)
            except BaseException as e:
                logger.error(f"agent loop exception: {e}")
                await app.send(
                    "Someone tell [CQ:at,qq=1571224208] there is a problem with my AI."
                )
                await app.send(str(e), target=CON)

    return agent_loop_wrapper


def main():
    App().run()


if __name__ == "__main__":
    main()
