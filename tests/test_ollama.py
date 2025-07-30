import logging

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

load_dotenv()


logging.getLogger("httpx").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class CityLocation(BaseModel):
    city: str
    country: str


ollama_model = OpenAIModel(model_name="qwen3:8b", provider=OpenAIProvider(base_url="http://localhost:11434/v1"))
agent = Agent(ollama_model, output_type=CityLocation)

result = agent.run_sync("Where were the olympics held in 2012?")
print(result.output)
# > city='London' country='United Kingdom'
print(result.usage())
# > Usage(requests=1, request_tokens=57, response_tokens=8, total_tokens=65)
