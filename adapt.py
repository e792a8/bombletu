import traceback
from langchain_core.embeddings import Embeddings
import asyncio
import httpx
from config import *

logger = get_log(__name__)


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
