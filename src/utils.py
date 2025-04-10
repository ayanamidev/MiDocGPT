import io
import logging
from typing import Generator, Tuple

import fitz
from PIL import Image
from PyPDF2 import PdfReader

from llm import LLM
from llm_utils import ImageMessageBuilder, LLMType, TextMessageBuilder

logger = logging.getLogger(__name__)



def pdf_bytes_to_images(
    pdf_bytes: io.BytesIO, dpi: int = 300
) -> Generator[Tuple[int, bytes], None, None]:
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)
            pix = page.get_pixmap(dpi=dpi)
            img_data = pix.samples
            img = Image.frombytes("RGB", [pix.width, pix.height], img_data)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()
            yield (page_number + 1, img_bytes)
        pdf_document.close()
    except Exception as e:
        logger.error(f"Error transforming pdf_bytes to screenshots: {str(e)}")
        raise e



class VisionModel:
    def __init__(
        self,
        llm_config_file: str = "src/llm_config.yaml",
        system_prompt_path: str = "src/pdf_to_markdown_system_prompt.txt",
    ):
        """
        Initializes the VisionModel with the specified configuration and system prompt.

        Args:
            llm_config_file (str): The path to the configuration file for the LLM.
            system_prompt_path (str): The path to the system prompt file to be used with the LLM.
        """
        self.llm = LLM(llm_type=LLMType.OPENAI, config_path=llm_config_file)
        self.system_prompt = open(system_prompt_path).read().strip()

    def send_image_and_text_to_gpt4_vision(
        self, img: bytes, extracted_text: str, page_num: int
    ) -> str:
        """
        Sends the extracted image and text to the GPT-4 vision model for processing.

        Args:
            img (bytes): The image data in bytes to be sent to the model.
            extracted_text (str): The extracted text from the PDF to be included in the request.

        Returns:
            str: The response text from the GPT-4 vision model.

        Raises:
            Exception: If there is an error during the model's response processing.
        """
        self.llm.message_builder = TextMessageBuilder()
        system_message = self.llm.create_message(
            content=self.system_prompt, role="system", llm_client=self.llm.llm_client
        )
        user_text_message = self.llm.create_message(
            content=f"The extracted text is: \n\n {extracted_text}",
            role="user",
            llm_client=self.llm.llm_client,
        )
        self.llm.message_builder = ImageMessageBuilder()
        user_image_message = self.llm.create_message(
            content=None,
            image_bytes=img,
            content_type="image",
            role="user",
            llm_client=self.llm.llm_client,
        )
        messages = [system_message, user_text_message, user_image_message]
        try:
            llm_response = self.llm.complete(messages)
            text_response = llm_response.text_response
            final_response = f"---- Begin Page {page_num}\n\n{text_response}\n\n ---- End Page {page_num}"
            return final_response
        except Exception as e:
            logger.error(f"Error during GPT-4 Vision processing: {e}")
            raise e
