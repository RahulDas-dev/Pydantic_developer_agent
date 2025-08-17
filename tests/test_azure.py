# ruff: noqa: T201
import asyncio
import logging
import os

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.azure import AzureProvider

load_dotenv()

logger = logging.getLogger(__name__)


region_name = os.environ.get("AWS_REGION", None)
aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID", None)
aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY", None)

model_name = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"


model_ = OpenAIModel(
    "finaclegpt4.1",
    provider=AzureProvider(
        openai_client=AsyncAzureOpenAI(
            azure_endpoint=os.environ.get("AZURE_API_BASE", ""),
            azure_deployment=model_name,
            api_version=os.environ.get("AZURE_API_VERSION", ""),
            api_key=os.environ.get("AZURE_API_KEY", ""),
        )
    ),
)
agent = Agent[None, str](
    name="Agent1",
    model=model_,
    output_type=str,
    retries=3,
    system_prompt="You are a helpful Abent help user to resolve their queries.",
)


async def run_agent(task: str) -> None:
    results = await agent.run(user_prompt=task)
    print(results)


asyncio.run(run_agent("what is the color of the sky??"))
