from typing import List

from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document


class Chunker:
    """
    A class that handles various methods of splitting or chunking text, such as recursive chunking
    and markdown-based chunking, to generate structured documents.
    """

    def __init__(self) -> None:
        pass

    def markdown_chunking(self, markdown_content: str) -> List[Document]:
        """
        Applies markdown header-based chunking to the given markdown content.

        Args:
            markdown_content (str): The markdown content to be split.

        Returns:
            List[Document]: A list of documents generated from the markdown text chunks.
        """

        """
            Esto indica que se va a cortar el texto cada vez que aparezca un:

            # (título principal)
            ## (subtítulo)
            ### (subsubtítulo)++++++++++++++++

        """
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, strip_headers=False
        )
        return markdown_splitter.split_text(markdown_content)

