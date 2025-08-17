# ruff: noqa: PLC0415
import os

from pydantic_ai.models import Model


def _build_azure_model(model_name: str) -> Model:
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


def _build_openai_model(model_name: str) -> Model:
    from openai import AsyncOpenAI
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.openai import OpenAIProvider

    # OPENAI_API_KEY  is required for OpenAI models

    return OpenAIModel(
        model_name,
        provider=OpenAIProvider(openai_client=AsyncOpenAI()),
    )


def _build_google_model(model_name: str, provider: str = "google-vertex") -> Model:
    from pydantic_ai.models.google import GoogleModel

    # VERTEXAI_PROJECT and VERTEXAI_LOCATION1 are required for Google Vertex AI models

    return GoogleModel(model_name=model_name, provider=provider)


def _build_anthropic_model(model_name: str, provider: str) -> Model:
    from pydantic_ai.models.anthropic import AnthropicModel

    return AnthropicModel(model_name, provider)


def _build_bedrock_model(model_name: str) -> Model:
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


def _build_ollama_model(model_name: str) -> Model:
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.openai import OpenAIProvider

    ollama_base_url = os.environ.get("OLLAMA_API_BASE", None)

    if ollama_base_url is None:
        raise ValueError("Environment variable OLLAMA_API_BASE must be set")

    return OpenAIModel(model_name, provider=OpenAIProvider(base_url=ollama_base_url))


def llm_factory() -> Model:
    provider_name = os.environ.get("ACTIVE_PROVIDER", None)
    model_name = os.environ.get("ACTIVE_MODEL", None)

    if provider_name is None or model_name is None:
        raise ValueError("Environment variable provider_name and model_name must be set")
    if provider_name == "azure":
        return _build_azure_model(model_name)

    if provider_name == "openai":
        return _build_openai_model(model_name)

    if provider_name == "google":
        return _build_google_model(model_name)

    if provider_name == "anthropic":
        return _build_anthropic_model(model_name, provider="anthropic")

    if provider_name == "aws":
        return _build_bedrock_model(model_name)

    if provider_name == "ollama":
        return _build_ollama_model(model_name)

    raise ValueError("Invalid client type provided")
