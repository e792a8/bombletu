from typing import List
from ncatbot.core import BotClient, MessageArray
from ncatbot.core.event import GroupMessageEvent
from asyncio_channel import create_channel, create_sliding_buffer
from agenting import BotContext, BotState, make_agent, make_chroma
import asyncio
from datetime import datetime
from pytz import timezone
from ncatbot.core.api import NapCatAPIError
import os, signal
from langchain.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_chroma import Chroma
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.memory import InMemorySaver
from time import time
from msgfmt import msglfmt, parse_msg
import traceback
from config import *
from app import App

logger = get_log(__name__)


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


async def agent_loop(
    app: App,
    agent: CompiledStateGraph[BotState, BotContext],
    agentconfig: RunnableConfig,
):
    msg_inject = []
    while True:
        logger.info("agent invoking")
        ret = await agent.ainvoke(
            {"messages": msg_inject, "idle_minutes": None, "memory": None},
            config=agentconfig,
            context=BotContext(app),
            print_mode="updates",
        )
        idle_mins = ret["idle_minutes"]
        logger.debug(f"agent return: {ret}")
        if idle_mins is not None:
            logger.info(f"agent sleeping {idle_mins}min")
        else:
            logger.info(f"agent continuing")
        intr = await app.wait_intr(idle_mins or 0)
        unread = await app.collect_unread()
        msg_inject = []
        info_inject = []
        if idle_mins is not None:
            if intr:
                info_inject.append("[idle interrupted]")
            else:
                info_inject.append("[idle finished]")
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
            logger.info("agent loop starting")
            try:
                # async with AsyncSqliteSaver.from_conn_string(
                #     DATADIR + "/ckpt.sqlite"
                # ) as ckptr:
                ckptr = InMemorySaver()
                chroma = make_chroma(GRP, DATADIR + "/chroma")
                agent = make_agent(ckptr)
                agentconfig = RunnableConfig(
                    configurable={
                        "thread_id": "1",
                        "app": app,
                        "qapi": app.qbot.api,
                        "chroma": chroma,
                    }
                )
                await agent_loop(app, agent, agentconfig)
            except BaseException as e:
                logger.error(f"agent loop exception: {traceback.format_exc()}")
                await app.qbot.api.send_group_text(
                    GRP,
                    "Someone tell [CQ:at,qq=1571224208] there is a problem with my AI.",
                )
                await app.qbot.api.send_group_text(
                    CON, f"{GRP} {traceback.format_exc()}"
                )

    return agent_loop_wrapper


def main():
    App(make_agent_loop, make_chroma(GRP, DATADIR + "/chroma")).run()


if __name__ == "__main__":
    main()
