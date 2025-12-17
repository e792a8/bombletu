from typing import override
from .base import BaseApplet
from langchain_mcp_adapters.sessions import Connection
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.tools import BaseTool
from mcp.client.session import ClientSession
from mcp.types import TextContent
from logging import getLogger

logger = getLogger(__name__)


class MCPApplet(BaseApplet):
    def __init__(self, name: str, mcp: ClientSession) -> None:
        self._name = name
        self._mcp = mcp

    @property
    @override
    def name(self) -> str:
        return self._name

    @override
    async def get_instructions(self):
        content = (await self._mcp.get_prompt("instructions")).messages[0].content
        assert isinstance(content, TextContent)
        return content.text

    @override
    async def get_tools(self) -> list[BaseTool]:
        return await load_mcp_tools(self._mcp)

    @override
    async def poll_events(
        self, timeout: float = 0, select: list[str] = []
    ) -> str | None:
        from time import time

        # MCP Resources 支持客户端订阅更新，也许可以方便改造主动推送
        # 但是可能不方便传参数
        # ret = await self._mcp.get_resources(
        #     self._name,
        #     uris=[f"applet://events?timeout={timeout}&select={','.join(select)}" for ev in select] if select else ["event://all"],
        # )
        until = time() + timeout
        while True:
            tim = min(60, max(0, until - time()))
            ret = (
                await self._mcp.get_prompt(
                    "events",
                    {"timeout": str(tim), "select": ",".join(select)},
                )
            ).messages
            if len(ret) > 0:
                assert isinstance(ret[0].content, TextContent)
                return ret[0].content.text
            if time() > until:
                return None

    @override
    async def get_status(self) -> str:
        content = (await self._mcp.get_prompt("status")).messages[0].content
        assert isinstance(content, TextContent)
        return content.text
