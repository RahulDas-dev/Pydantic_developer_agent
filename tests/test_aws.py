# ruff: noqa: T201
import asyncio
import logging
import os

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.providers.bedrock import BedrockProvider

load_dotenv()

logger = logging.getLogger(__name__)


region_name = os.environ.get("AWS_REGION", None)
aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID", None)
aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY", None)

model_name = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"

model_ = BedrockConverseModel(
    model_name,
    provider=BedrockProvider(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name,
        profile_name=os.environ.get("AWS_PROFILE", None),
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
