from abc import ABC, abstractmethod
from langchain_core.tools import BaseTool


class BaseApplet(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    async def get_instructions(self) -> str | None:
        return None

    async def get_tools(self) -> list[BaseTool]:
        return []

    async def poll_events(
        self, timeout: float = 0, select: list[str] = []
    ) -> str | None:
        return None

    async def get_status(self) -> str | None:
        return None
