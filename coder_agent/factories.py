# ruff: noqa: PLC0415
import os
from typing import Literal

from dotenv import load_dotenv
from pydantic_ai.models import Model

load_dotenv()

TClientName = Literal["azure", "openai", "google", "anthropic", "aws"]
TModelName = Literal[
    "gpt-4.0",
    "gpt-4",
    "gpt-4o",
    "gemini-2.5-pro",
    "claude-3-opus20240229",
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0",  # This is BEDROCK MODEL
]


def llm_factory(provider_name: TClientName, model_name: TModelName | None = None) -> Model:
    if provider_name == "azure":
        from openai import AsyncAzureOpenAI
        from pydantic_ai.models.openai import OpenAIModel
        from pydantic_ai.providers.azure import AzureProvider

        return OpenAIModel(
            model_name,
            provider=AzureProvider(
                openai_client=AsyncAzureOpenAI(
                    azure_endpoint=os.environ.get("AZURE_API_BASE", ""),
                    azure_deployment=model_name,
                    api_version=os.environ.get("AZURE_API_VERSION", ""),
                    api_key=os.environ.get("AZURE_API_KEY", ""),
                )
            ),
        )

    if provider_name == "openai":
        from openai import AsyncOpenAI
        from pydantic_ai.models.openai import OpenAIModel
        from pydantic_ai.providers.openai import OpenAIProvider

        return OpenAIModel(model_name, provider=OpenAIProvider(openai_client=AsyncOpenAI()))

    if provider_name == "google":
        from pydantic_ai.models.gemini import GeminiModel

        model_name = model_name or "gemini-2.5-pro"
        if model_name not in ["gemini-2.0-flash-001", "gemini-2.5-pro"]:
            raise ValueError("Invalid model name for Google Vertex AI")

        return GeminiModel(model_name=model_name, provider="google-vertex")

    if provider_name == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel

        project_id = os.environ.get("VERTEXAI_PROJECT", None)
        region = os.environ.get("VERTEXAI_LOCATION1", None)
        if project_id is None or region is None:
            raise ValueError("Environment variables VERTEXAI_PROJECT and VERTEXAI_LOCATION1 must be set")

        return AnthropicModel(
            "claude-3-opus20240229",
        )

    if provider_name == "aws":
        from pydantic_ai.models.bedrock import BedrockConverseModel
        from pydantic_ai.providers.bedrock import BedrockProvider

        region_name = os.environ.get("AWS_REGION", None)
        aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID", None)
        aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY", None)

        if region_name is None or aws_access_key_id is None or aws_secret_access_key is None:
            raise ValueError("Environment variable AWS REGION must be set")

        model_name = model_name or "us.anthropic.claude-3-7-sonnet-20250219-v1:0"

        return BedrockConverseModel(
            model_name,
            provider=BedrockProvider(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name,
            ),
        )

    raise ValueError("Invalid client type provided")
