from collections.abc import Callable
import traceback
from typing import Dict, Sequence
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models import LanguageModelLike, BaseChatModel
import asyncio
import httpx
from langchain_core.tools import BaseTool
from config import *

logger = get_log(__name__)


class ChatMux(BaseChatModel):
    def __init__(self, models: list[LanguageModelLike]):
        self._models = models

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return ChatMux(
            [m.bind_tools(tools, tool_choice=tool_choice) for m in self._models]  # type: ignore
        )

    def invoke(self, input, config=None, *, stop=None, **kwargs) -> AIMessage:
        raise NotImplementedError

    async def ainvoke(self, input, config=None, **kwargs) -> AIMessage:
        pending = [
            asyncio.create_task(m.ainvoke(input, config, **kwargs))
            for m in self._models
        ]
        logger.info(f"requesting {len(self._models)} llms")
        result = None
        exc = []
        while True:
            if not pending:
                break
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for d in done:
                logger.info(f"done response: {d}")
                if e := d.exception():
                    exc.append(e)
                else:
                    result = d.result()
                    break
            if result is not None:
                break
        for p in pending:
            p.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if result is not None:
            return result  # type: ignore
        else:
            raise ExceptionGroup(f"all {len(self._models)} llm invokes failed", exc)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        raise NotImplementedError

    @property
    def _llm_type(self) -> str:
        return "chat-mux"


class GiteeAIEmbeddings(Embeddings):
    def __init__(
        self, model: str, base_url: str, api_key: str, dimensions: int
    ) -> None:
        from lru import LRU

        super().__init__()
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.dimensions = dimensions
        self.lru = LRU(120)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return asyncio.run(self.aembed_documents(texts))

    def embed_query(self, text: str) -> list[float]:
        return asyncio.run(self.aembed_query(text))

    async def aembed_query(self, text: str) -> list[float]:
        return (await self.aembed_documents([text]))[0]

    # def _s_request(self, client: httpx.Client, t: str) -> list[float]:
    #     if res := self.lru.get(t):
    #         return res
    #     for retry in range(3):
    #         try:
    #             resp = client.post(
    #                 "https://ai.gitee.com/v1/embeddings",
    #                 timeout=30,
    #                 headers={
    #                     "Content-Type": "application/json",
    #                     "Authorization": "Bearer " + self.api_key,
    #                 },
    #                 json={
    #                     "model": self.model,
    #                     "input": t,
    #                     "encoding_format": "float",
    #                     "dimensions": self.dimensions,
    #                 },
    #             )
    #             res = resp.json()["data"][0]["embedding"]
    #             self.lru[t] = res
    #             return res
    #         except Exception as e:
    #             logger.error(f"embedder error: {traceback.format_exc()}")
    #             time.sleep(retry * 5 + 5)
    #     raise Exception("embedder: too many retries")

    async def _request(self, client: httpx.AsyncClient, t: str) -> list[float]:
        if res := self.lru.get(t):
            return res
        for retry in range(3):
            try:
                resp = await client.post(
                    "https://ai.gitee.com/v1/embeddings",
                    timeout=30,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": "Bearer " + self.api_key,
                    },
                    json={
                        "model": self.model,
                        "input": t,
                        "encoding_format": "float",
                        "dimensions": self.dimensions,
                    },
                )
                res = resp.json()["data"][0]["embedding"]
                self.lru[t] = res
                return res
            except Exception:
                logger.error(f"embedder error: {traceback.format_exc()}")
                await asyncio.sleep(retry * 5 + 5)
        raise Exception("embedder: too many retries")

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient() as client:
            ret = await asyncio.gather(*(self._request(client, t) for t in texts))
        return ret
