import logging
from typing import List, Optional

import dotenv
import yaml

from llm_utils import (
    CompletionStrategy,
    ImageMessageBuilder,
    LLMClientFactory,
    LLMResponse,
    LLMType,
    MessageBuilder,
    ModelPricingManager,
    NonOpenAICompletionStrategy,
    OpenAICompletionStrategy,
    TextMessageBuilder,
)

dotenv.load_dotenv()
logger = logging.getLogger(__name__)


class LLM:
    """Clase para interactuar con un modelo de lenguaje (LLM)."""

    def __init__(self, llm_type: LLMType, config_path: str = "src/llm/config.yaml"):
        """
        Inicializa el cliente LLM y carga la configuración.

        Args:
            llm_type (LLMType): El tipo de cliente LLM a usar.
            config_path (str): La ruta al archivo de configuración YAML.
        """
        # Carga la configuración del archivo YAML
        with open(config_path, "r") as config_file:
            self.config = yaml.safe_load(config_file)

        self.model_name = self.config["model"]

        # Creación del cliente LLM a través de la fábrica
        self.llm_client = LLMClientFactory.create_client(
            llm_type, self.model_name, self.config
        )

        # Selección de la estrategia de completion
        self.completion_strategy = self._select_completion_strategy(llm_type)

        # Selección del creador de mensajes
        self.message_builder = self._select_message_builder()

        self.pricing_manager = ModelPricingManager()
        self.conversation_history = []

    def _select_message_builder(self) -> MessageBuilder:
        """Selecciona el creador de mensajes adecuado basado en el tipo de contenido."""
        content_type = self.config.get("default_content_type", "text")
        if content_type == "text":
            return TextMessageBuilder()
        elif content_type == "image":
            return ImageMessageBuilder()
        else:
            raise ValueError(f"Unsupported content type: {content_type}")

    def _select_completion_strategy(self, llm_type: LLMType) -> CompletionStrategy:
        """Selecciona la estrategia de completion adecuada según el tipo de LLM."""
        if llm_type != LLMType.NOT_OPENAI:
            return OpenAICompletionStrategy()
        else:
            return NonOpenAICompletionStrategy()

    def create_message(
        self, content: Optional[str], llm_client, role: str = "user", **kwargs
    ) -> dict:
        """
        Crea un mensaje delegando la tarea al creador de mensajes adecuado.

        Args:
            content (Optional[str]): El contenido del mensaje.
            role (str): El rol del remitente del mensaje ('user' o 'assistant').
            kwargs: Parámetros adicionales como image_url, image_path, etc.

        Returns:
            dict: El mensaje creado en formato de diccionario.
        """
        return self.message_builder.create_message(
            role=role, llm_client=llm_client, content=content, **kwargs
        )

    def complete(self, messages: list) -> LLMResponse:
        """
        Usa la estrategia de completion adecuada para generar una respuesta del modelo.

        Args:
            messages (list): La lista de mensajes enviados al modelo.

        Returns:
            LLMResponse: La respuesta del modelo con detalles como el costo, los tokens usados, etc.
        """
        return self.completion_strategy.complete(self.llm_client, messages, self.config)

    def chat(self, content: str) -> LLMResponse:
        """Método principal para manejar el chat conversacional.

        Args:
            content (str): El contenido del mensaje del usuario.

        Returns:
            LLMResponse: Un objeto que contiene la respuesta del modelo, junto con los conteos de tokens.
        """
        user_message = self.create_message(
            content=content, role="user", llm_client=self.llm_client
        )
        self.conversation_history.append(user_message)

        response = self.complete(self.conversation_history)

        assistant_message = self.create_message(
            content=response.text_response, role="assistant", llm_client=self.llm_client
        )
        self.conversation_history.append(assistant_message)

        return response

    def clear_conversation_history(self):
        """Limpia el historial de conversación."""
        self.conversation_history = []

    def get_conversation_history(self) -> List[dict]:
        """Devuelve el historial de conversación.

        Returns:
            List[dict]: Una lista de diccionarios que representan el historial de la conversación.
        """
        return self.conversation_history
