from time import sleep
from applet import MCPApplet
from config import Q_MCP_PORT
from subprocess import Popen
import asyncio


class OicqApplet(MCPApplet):
    def __init__(self) -> None:
        self.proc = Popen(["python", "-m", "oicq.mcp", Q_MCP_PORT])

        super().__init__(
            "oicq",
            {
                "transport": "streamable_http",
                "url": f"http://127.0.0.1:{Q_MCP_PORT}/mcp",
            },
        )

        for _ in range(10):
            try:
                asyncio.run(self.get_tools())
            except BaseException:
                sleep(1)
        asyncio.run(self.get_tools())

    def __del__(self):
        self.proc.terminate()
