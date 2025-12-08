from typing import override
from .base import BaseApplet
from langchain_mcp_adapters.sessions import Connection
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool


class MCPApplet(BaseApplet):
    def __init__(self, name: str, mcpconfig: Connection) -> None:
        self._name = name
        self._mcp = MultiServerMCPClient({name: mcpconfig})
        self._mcp.get_tools

    @property
    @override
    def name(self) -> str:
        return self._name

    @override
    async def get_instructions(self):
        return (await self._mcp.get_prompt(self._name, "instructions"))[0].text

    @override
    async def get_tools(self) -> list[BaseTool]:
        return await self._mcp.get_tools()

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
            ret = await self._mcp.get_prompt(
                self._name,
                "events",
                arguments={"timeout": str(tim), "select": ",".join(select)},
            )
            if len(ret) > 0:
                return ret[0].text
            if time() > until:
                return None

    @override
    async def get_status(self) -> str:
        return (await self._mcp.get_prompt(self._name, "status"))[0].text
