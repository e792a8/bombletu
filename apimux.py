from fastapi import FastAPI, Request, Response
import uvicorn
import httpx
import asyncio
import json
from os import environ
from dotenv import load_dotenv
from logging import getLogger
import logging

load_dotenv()
logger = getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

fapi = FastAPI()
client = httpx.AsyncClient()


def collect_llms():
    llms = []
    for i in range(1, 100):
        if environ.get(f"LLM{i}_MODEL"):
            llms.append(
                {
                    "model": environ.get(f"LLM{i}_MODEL"),
                    "api_key": environ.get(f"LLM{i}_API_KEY"),
                    "base_url": environ.get(f"LLM{i}_BASE_URL"),
                }
            )
    return llms


llms = collect_llms()


async def request_llm(request: Request, llm: dict):
    method = request.method
    path = request.path_params["path"]
    body = await request.json() if await request.body() else None
    logger.info(f"requesting {len(llms)} models")
    headers = httpx.Headers()
    headers["Authorization"] = f"Bearer {llm['api_key']}"
    headers["Content-Type"] = "application/json"
    if isinstance(body, dict):
        body["model"] = llm["model"]
    resp = await client.request(
        method,
        llm["base_url"].rstrip("/") + "/" + path,
        json=body,
        headers=headers,
        timeout=60,
    )
    await resp.aread()
    if resp.status_code != 200 or resp.json().get("error"):
        logger.error(
            f"request for llm {llm} error: {resp.status_code} {resp.content.decode()}"
        )
        return None
    ret = resp.json()
    return ret


@fapi.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def api_v1(request: Request):
    pending = []
    logger.info(f"requesting {len(llms)} models")
    for llm in llms:
        task = request_llm(request, llm)
        pending.append(asyncio.create_task(task))
    result = None
    exc = []
    while True:
        if not pending:
            break
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
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
    if result is not None:
        return result
    else:
        return Response(
            json.dumps(
                {
                    "error": {
                        "code": "503",
                        "message": f"all {len(llms)} llm invokes failed",
                    }
                }
            ),
            status_code=503,
        )


def main():
    from sys import argv

    host = argv[1] if len(argv) > 1 else "127.0.0.1"
    port = int(argv[2]) if len(argv) > 2 else 17412
    uvicorn.run(fapi, host=host, port=port)


if __name__ == "__main__":
    main()
