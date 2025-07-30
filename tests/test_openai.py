# ruff: noqa: T201
import asyncio
import logging
from dataclasses import dataclass
from string import Template
from typing import Union

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ToolReturn
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

load_dotenv()

logger = logging.getLogger(__name__)


class Failure(BaseModel):
    """Represents structure for action failure"""

    reason: str = Field(description="The Reason why the task failed")


@dataclass(frozen=True, slots=True)
class AgentContext:
    shop_name: str


model_ = OpenAIModel(
    model_name="gpt-4o-mini",
    provider=OpenAIProvider(openai_client=AsyncOpenAI()),
)

agent = Agent[AgentContext, str | Failure](
    name="Fruit Shop Agent",
    model=model_,
    deps_type=AgentContext,
    output_type=Union[str, Failure],  # type: ignore # noqa: PGH003
    retries=3,
)

SYSTEM_MESSAGE = (
    "You are a helpful Food Seller agent.\n"
    "You help users with information about food items, their colors, and other details.\n"
    "You are working at the shop: ${SHOP_NAME}"
)


@agent.system_prompt
async def get_system_prompt(ctx: RunContext[AgentContext]) -> str:
    # The system prompt uses the shop_name from AgentContext
    return Template(SYSTEM_MESSAGE).substitute(SHOP_NAME=ctx.deps.shop_name)


@agent.tool
async def get_fruit_price(ctx: RunContext[AgentContext], food_name: str) -> ToolReturn:
    """Get the price of a fruit per kilogram.

    Args:
        food_name: The name of the fruit.

    Returns:
        returns price of the fruit or an error message if not found.
    """
    food_prices = {
        "apple": 1.2,
        "banana": 0.5,
        "grape": 2.0,
        "pineapple": 3.0,
        "orange": 1.0,
        "kiwi": 1.5,
        "mango": 2.5,
        "strawberry": 2.0,
    }

    price = food_prices.get(food_name.lower(), None)
    if price is None:
        return ToolReturn(
            return_value=None,
            content=f"Price of {food_name} is not known.",
            metadata={"success": False, "error": "UNKNOWN_FRUIT"},
        )
    return ToolReturn(
        return_value=price,
        content=f"The price of {food_name} is {price} per kilogram.",
        metadata={"success": True},
    )


task_str = "what is the price of an apple, banana, grape, pineapple?"


async def run_agent(shop_name: str, task: str) -> None:
    context = AgentContext(shop_name=shop_name)
    results = await agent.run(user_prompt=task, deps=context)
    print(results)


shop_name = "Rahul's Fresh Foods"
asyncio.run(run_agent(shop_name, task_str))
