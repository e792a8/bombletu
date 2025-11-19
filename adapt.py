from langchain_core.embeddings import Embeddings
import asyncio
import httpx


class GiteeAIEmbeddings(Embeddings):
    def __init__(
        self, model: str, base_url: str, api_key: str, dimensions: int
    ) -> None:
        super().__init__()
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return asyncio.run(self.aembed_documents(texts))

    def embed_query(self, text: str) -> list[float]:
        return asyncio.run(self.aembed_query(text))

    async def aembed_query(self, text: str) -> list[float]:
        return (await self.aembed_documents([text]))[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient() as client:
            ret = []
            for t in texts:
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
                ret.append(resp.json()["data"][0]["embedding"])
        return ret
