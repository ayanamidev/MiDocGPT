import base64
import io
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage as NotOpenAISystemMessage
from azure.ai.inference.models import UserMessage as NotOpenAIUserMessage
from azure.core.credentials import AzureKeyCredential
from langchain_openai import AzureChatOpenAI as LangchainAzureOpenAI
from llama_index.core.llms import ChatMessage as LlamaIndexChatMessage
from llama_index.llms.azure_openai import AzureOpenAI as LlamaIndexAzureOpenAI
from openai import AzureOpenAI
from PIL import Image

# Create a logger object for logging error messages
logger = logging.getLogger(__name__)


class LLMType(Enum):
    OPENAI = "openai"
    LANGCHAIN = "langchain"
    LLAMA_INDEX = "llama_index"
    NOT_OPENAI = "not_openai"


@dataclass
class LLMResponse:
    """Data class for storing the response from the language model.

    Attributes:
        text_response (str): The generated text response from the model.
        completion_tokens (int): The number of tokens used in the completion.
        prompt_tokens (int): The number of tokens used in the prompt.
        cost (Optional[float]): The cost of the response, if calculated.
        time_taken (Optional[float]): The time taken to generate the response.
    """

    text_response: str
    completion_tokens: int
    prompt_tokens: int
    system_prompt: str
    user_prompt: str
    model: Optional[str] = None
    cost: Optional[float] = None
    time_taken: Optional[float] = None


@dataclass(frozen=True)
class ModelPricing:
    """Data class to store model-specific pricing information.

    Attributes:
        model_name (str): The name of the model.
        input_token_price (float): The price per input token.
        output_token_price (float): The price per output token.
    """

    model_name: str
    input_token_price: float
    output_token_price: float


class ModelPricingManager:
    """Class to manage pricing for predefined models.

    This class contains predefined pricing for various models and allows lookup by model name.
    """

    def __init__(self):
        """Initializes the manager with predefined model pricing."""
        self.models_pricing = {
            "gpt-4o-mini": ModelPricing(
                "gpt-4o-mini", input_token_price=0.15, output_token_price=0.6
            ),
            "gpt-4o": ModelPricing(
                "gpt-4o", input_token_price=2.50, output_token_price=10.00
            ),
        }

    def get_model_pricing(self, model_name: str) -> Optional[ModelPricing]:
        """Retrieves the pricing for a specific model by name.

        Args:
            model_name (str): The name of the model.

        Returns:
            Optional[ModelPricing]: The pricing information for the model, or None if not found.
        """
        return self.models_pricing.get(model_name)


class LLMClientFactory:
    """
    Factory para crear clientes LLM basados en el tipo de modelo.
    Este patrón desacopla la lógica de inicialización de los clientes de la clase principal LLM.
    """

    @staticmethod
    def create_client(llm_type: LLMType, model_name: str, config: dict):
        """Crea y retorna el cliente LLM apropiado basado en el tipo de LLM."""
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        api_version = os.environ.get("OPENAI_API_VERSION")

        model_name = config["model"]

        if model_name == "gpt-4o":
            azure_endpoint = os.environ.get("AZURE_OPENAI_GPT_4O_ENDPOINT")
        elif model_name == "gpt-4o-mini":
            azure_endpoint = os.environ.get("AZURE_OPENAI_GPT_4O_MINI_ENDPOINT")
        else:
            logger.info("Model not supported")

        if llm_type == LLMType.OPENAI:
            return AzureOpenAI(
                api_key=api_key, api_version=api_version, azure_endpoint=azure_endpoint
            )
        elif llm_type == LLMType.LANGCHAIN:
            return LangchainAzureOpenAI(
                azure_deployment=model_name,
                openai_api_version=api_version,
                azure_endpoint=azure_endpoint,
                api_key=api_key,
            )
        elif llm_type == LLMType.LLAMA_INDEX:
            return LlamaIndexAzureOpenAI(
                model=model_name,
                engine=model_name,
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
            )
        elif llm_type == LLMType.NOT_OPENAI:
            return ChatCompletionsClient(
                endpoint=os.environ.get("AZURE_CHAT_COMPLETIONS_ENDPOINT"),
                credential=AzureKeyCredential(os.getenv("AZURE_CHAT_COMPLETIONS_KEY")),
            )
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")


# Creadores de mensajes parametrizados
class MessageBuilder(ABC):
    """
    Clase abstracta para la creación de mensajes basados en diferentes tipos de contenido.
    Esto permite la creación flexible de distintos tipos de mensajes (texto, imágenes, etc.).
    """

    @abstractmethod
    def create_message(
        self, role: str, llm_client, content: Optional[str], **kwargs
    ) -> dict:
        """Método abstracto para crear un mensaje basado en el tipo de contenido."""
        pass


class CompletionStrategy(ABC):
    """
    Clase base abstracta para manejar las diferentes estrategias de completion.
    Cada subclase implementará una estrategia diferente según el tipo de modelo (OpenAI, LlamaIndex, etc.).
    """

    @abstractmethod
    def complete(self, llm_client, messages: list, config: dict) -> LLMResponse:
        """Método abstracto para manejar la generación de respuestas."""
        pass


class TextMessageBuilder(MessageBuilder):
    """Clase concreta para construir mensajes de texto."""

    def create_message(
        self, role: str, llm_client, content: Optional[str], **kwargs
    ) -> dict:
        if content is None:
            raise ValueError("Content cannot be None for text messages")
        if isinstance(llm_client, LlamaIndexAzureOpenAI):
            return LlamaIndexChatMessage(role=role, content=content)
        return {"role": role, "content": [{"type": "text", "text": content}]}


class ImageMessageBuilder(MessageBuilder):
    """Clase concreta para construir mensajes con imágenes."""

    def create_message(
        self, role: str, llm_client, content: Optional[str], **kwargs
    ) -> dict:
        image_url = kwargs.get("image_url")
        image_path = kwargs.get("image_path")
        image_bytes = kwargs.get("image_bytes")

        if image_url:
            image_data = image_url
        elif image_path:
            image_data = load_image_from_path(image_path)
        elif image_bytes:
            img_str = base64.b64encode(image_bytes).decode("utf-8")
            image_data = f"data:image/png;base64,{img_str}"
        else:
            raise ValueError("No valid image source provided (URL, path, or bytes).")

        content_list = [{"type": "image_url", "image_url": {"url": image_data}}]
        if content:
            content_list.insert(0, {"type": "text", "text": content})

        return {"role": role, "content": content_list}


class OpenAICompletionStrategy(CompletionStrategy):
    """Estrategia para manejar las completions en OpenAI."""

    def complete(self, llm_client, messages: list, config: dict) -> LLMResponse:
        # system_prompt = next(  # FIXME: arreglar esto para que no pete.
        #     (msg["content"] for msg in messages if msg["role"] == "system"), None
        # )
        # user_prompt = "\n".join(
        #     msg["content"] for msg in messages if msg["role"] == "user"
        # )
        start_time = time.time()
        if isinstance(
            llm_client, LangchainAzureOpenAI
        ):  # FIXME: creo que queda un poco fea esta manera de hacerlo.
            completion = llm_client.invoke(input=messages, **config)
            text_response = completion.content
            completion_tokens = completion.response_metadata["token_usage"][
                "completion_tokens"
            ]
            prompt_tokens = completion.response_metadata["token_usage"]["prompt_tokens"]
        elif isinstance(llm_client, AzureOpenAI):
            completion = llm_client.chat.completions.create(messages=messages, **config)
            text_response = completion.choices[0].message.content.strip()
            completion_tokens = completion.usage.completion_tokens
            prompt_tokens = completion.usage.prompt_tokens
        elif isinstance(llm_client, LlamaIndexAzureOpenAI):
            text_response = llm_client.chat(messages=messages, **config)
            completion_tokens = 0
            prompt_tokens = 0

        total_time = time.time() - start_time

        pricing = ModelPricingManager().get_model_pricing(model_name=config["model"])
        total_cost = (
            pricing.input_token_price * prompt_tokens
            + pricing.output_token_price * completion_tokens
        ) / 1_000_000

        return LLMResponse(
            text_response=text_response,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            cost=total_cost,
            time_taken=total_time,
            model=config["model"],
            system_prompt="",
            user_prompt="",
        )


class NonOpenAICompletionStrategy(CompletionStrategy):
    """Estrategia para manejar las completions en modelos no OpenAI (LlamaIndex, Langchain, etc.)."""

    def complete(self, llm_client, messages: list, config: dict) -> LLMResponse:
        system_prompt = next(
            (msg["content"] for msg in messages if msg["role"] == "system"), None
        )
        user_prompt = next(
            (msg["content"] for msg in messages if msg["role"] == "user"), ""
        )

        model_response = llm_client.complete(
            messages=[
                NotOpenAISystemMessage(content=system_prompt),
                NotOpenAIUserMessage(content=user_prompt),
            ],
            **config,
        )

        response = model_response.choices[0].message.content
        prompt_tokens = model_response.usage.prompt_tokens
        completion_tokens = model_response.usage.completion_tokens

        return LLMResponse(
            text_response=response,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            cost=None,  # Custom pricing logic can be added.
            time_taken=None,  # Custom time tracking logic.
            model=config["model"],
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )


def load_image_from_path(image_path: str, image_format: str = "PNG") -> Optional[str]:
    """
    Load an image from the specified local path and return its base64-encoded representation.

    Args:
        image_path (str): The file path to the image to be loaded.
        image_format (str, optional): The format in which to save the image. Default is "PNG".

    Returns:
        Optional[str]: A base64-encoded string representing the image. Returns None if an error occurs.
    """
    try:
        if not os.path.exists(image_path):
            logger.error(f"Image path does not exist: {image_path}")
            return None

        # Open the image file using the Pillow library (PIL)
        with Image.open(image_path) as img:
            buffered = io.BytesIO()
            img.save(buffered, format=image_format)
            img_str = base64.b64encode(buffered.getvalue()).decode()

            return f"data:image/png;base64,{img_str}"

    except Exception as e:
        logger.error(f"Error loading image: {e}")
        return None
