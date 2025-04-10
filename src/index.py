import os
import logging
import dotenv
import tqdm

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    VectorSearch,
    VectorSearchProfile,
)

from langchain_openai import AzureOpenAIEmbeddings

from llm import LLM
from chunking import Chunker

dotenv.load_dotenv()
logger = logging.getLogger(__name__)


class Indexer:
    """
    Clase para gestionar la indexación de documentos en Azure Search.

    Atributos:
        embeddings_instance (AzureOpenAIEmbeddings): Instancia de embeddings de Azure OpenAI.
        index_client (SearchIndexClient): Cliente para gestionar índices de búsqueda.
        search_client (SearchClient): Cliente para subir documentos al índice.
        document_loader (DocumentLoader): Cargador de documentos.
        chunker (Chunker): Herramienta para trocear documentos en fragmentos.
        index_name (str): Nombre del índice de búsqueda.
        chunking_method (str): Método para trocear documentos.
        llm (LLM): Modelo de lenguaje para el procesamiento contextual.
    """

    def __init__(self, index_name: str, chunking_method: str = "markdown") -> None:
        """
        Inicializa la clase Indexer con los clientes y herramientas necesarias.

        Args:
            index_name (str): Nombre del índice a crear o usar.
            chunking_method (str): Método para trocear documentos. Por defecto "markdown".
        """

        ##Azure Open AI
        ##Crea una instancia (una conexión lista para usar) que permite generar embeddings usando el modelo que tú desplegaste en Azure.
        self.embeddings_instance: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("EMBEDDINGS_DEPLOYMENT_NAME"),
            api_version=os.getenv("OPENAI_API_VERSION_EMBEDDINGS"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_EMBEDDINGS"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )

        ##Azure AI Search
        ## SearchIndexClient sirve para definir y gestionar la estructura de los índices.
        self.index_client = SearchIndexClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY")),
        )
        ## SearchClient sirve para consultar o modificar los documentos que están en un índice.
        self.search_client = SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            index_name=index_name,
            credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY")),
        )

        ##Clase que se encarga de leer archivos PDF, Word, Markdown, etc.
    

        ##Clase para trocear los documentos en fragmentos más pequeños (chunks), que luego se indexan.
        self.chunker = Chunker()

        ##Guarda el nombre del índice como atributo de la instancia
        self.index_name = index_name

        ## Guarda el método de troceado elegido (markdown, recursive, etc.)
        self.chunking_method = chunking_method
        
        ##Crea una instancia del modelo de lenguaje (LLM) que usarás para procesar contexto, si haces contextual_indexing.
        self.llm = LLM()

    def create_search_index(self) -> None:
        """
        Crea un índice de búsqueda en Azure Cognitive Search con configuración de vectorización y campos personalizados.
        """
        fields = [
            SearchField(name="parent_id", type=SearchFieldDataType.String),
            SearchField(name="grand_parent_id", type=SearchFieldDataType.String),
            SearchField(
                name="document_name",
                type=SearchFieldDataType.String,
                filterable=True,
                searchable=True,
            ),
            SearchField(
                name="chunk_id",
                type=SearchFieldDataType.String,
                key=True,
                retrievable=True,
                analyzer_name="keyword",
            ),
            SearchField(
                name="chunk",
                type=SearchFieldDataType.String,
                searchable=True,
                analyzer_name="standard.lucene",
            ),
            SearchField(
                name="chunk_metadata",
                type=SearchFieldDataType.String,
                searchable=True,
            ),
            SearchField(
                name="text_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                vector_search_dimensions=3072,
                vector_search_profile_name="myHnswProfile",
                searchable=True,
                retrievable=True,
            ),
        ]

        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="myHnsw")],
            profiles=[
                VectorSearchProfile(
                    name="myHnswProfile",
                    algorithm_configuration_name="myHnsw",
                    vectorizer_name="my_vectorizer",
                )
            ],
            vectorizers=[
                AzureOpenAIVectorizer(
                    vectorizer_name="my_vectorizer",
                    kind="azureOpenAI",
                    parameters=AzureOpenAIVectorizerParameters(
                        resource_url=os.getenv("AZURE_OPENAI_RESOURCE_URI"),
                        deployment_name=os.getenv("EMBEDDINGS_DEPLOYMENT_NAME"),
                        model_name=os.getenv("EMBEDDINGS_DEPLOYMENT_NAME"),
                    ),
                ),
            ],
        )

        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search,
        )

        result = self.index_client.create_or_update_index(index)
        logger.info(f"El índice '{result.name}' ha sido creado o actualizado correctamente.")

    def indexing(self, folder_path: str, load_pdf_method: str = "vision_model") -> None:
        """
        Indexa todos los documentos de una carpeta dividiéndolos en fragmentos y generando embeddings.

        Args:
            folder_path (str): Ruta a la carpeta con archivos a indexar.
            load_pdf_method (str): Método para procesar PDFs (por ejemplo: "document_intelligence").
        """
        try:
            ##Este bucle recorre todas las carpetas y subcarpetas dentro de folder_path.
            for root, _, files in os.walk(folder_path):
                ##Este recorre todos los archivos dentro de la carpeta actual (root).
                for filename in tqdm.tqdm(files, desc=f"Indexando archivos desde '{folder_path}'"):
                    ##Construye la ruta completa al archivo combinando
                    ##os.path.join(...) se encarga de colocar bien las barras / o \ según tu sistema operativo (Windows, Linux, Mac…).
                    file_path = os.path.join(root, filename)

                    if filename.endswith(".pdf"):
                        markdown_content = self.document_loader.load_pdf(file_path, method=load_pdf_method)
                    elif filename.endswith(".doc") or filename.endswith(".docx"):
                        markdown_content = self.document_loader.load_word(file_path)
                    elif filename.endswith(".html"):
                        markdown_content = self.document_loader.load_html(file_path)
                    elif filename.endswith(".md"):
                        markdown_content = open(file_path).read()
                    else:
                        logger.info(f"Formato no soportado: {filename}")
                        continue

                    logger.info(f"Archivo {filename} cargado correctamente.")
                    ##Genera un ID único para el documento usando hash.
                    document_id = hash(markdown_content)
                    ##Crea una lista vacía para almacenar los documentos que se van a indexar.
                    documents = []

                    if self.chunking_method == "markdown":
                        chunks = self.chunker.markdown_chunking(markdown_content)
                    elif self.chunking_method == "recursive":
                        chunks = self.chunker.recursive_chunking(markdown_content)
                    elif self.chunking_method == "markdown_and_recursive":
                        chunks = self.chunker.markdown_and_recursive_chunking(markdown_content)
                    else:
                        logger.error("Método de división de contenido no válido.")
                        continue

                    for chunk in chunks:
                        embedding = self.embeddings_instance.embed_documents([chunk.page_content])[0]
                        document = {
                            "chunk_id": f"chunk_{hash(chunk.page_content)}",
                            "parent_id": f"parent_chunk_{document_id}",
                            "grand_parent_id": "",
                            "document_name": filename,
                            "chunk": chunk.page_content,
                            "chunk_metadata": str(chunk.metadata),
                            "text_vector": embedding,
                        }
                        documents.append(document)

                    self.search_client.upload_documents(documents)
                    logger.info(f"{filename} indexado correctamente con {len(chunks)} fragmentos.")

        except Exception as e:
            logger.error(f"Error al indexar: {str(e)}")

    def contextual_indexing(self, folder_path: str, load_pdf_method: str = "vision_model") -> None:
        """
        Realiza una indexación contextual usando un modelo de lenguaje para enriquecer cada fragmento con contexto.

        Args:
            folder_path (str): Ruta a la carpeta con documentos a indexar.
            load_pdf_method (str): Método para cargar PDFs (por ejemplo: "document_intelligence").
        """
        logger.info("Usando indexación contextual.")
        system_prompt_path = "src/rag/index/prompts/contextual_indexing_system_prompt.txt"
        system_prompt = open(system_prompt_path).read().strip()
        total_indexing_cost = 0

        try:
            for root, _, files in os.walk(folder_path):
                for filename in tqdm.tqdm(files, desc=f"Indexando archivos desde '{folder_path}'"):
                    file_path = os.path.join(root, filename)

                    if filename.endswith(".pdf"):
                        markdown_content = self.document_loader.load_pdf(file_path, method=load_pdf_method)
                    elif filename.endswith(".doc") or filename.endswith(".docx"):
                        markdown_content = self.document_loader.load_word(file_path)
                    elif filename.endswith(".html"):
                        markdown_content = self.document_loader.load_html(file_path)
                    elif filename.endswith(".md"):
                        markdown_content = open(file_path).read()
                    else:
                        logger.info(f"Formato no soportado: {filename}")
                        continue

                    chunks = self.chunker.markdown_chunking(markdown_content)
                    document_id = hash(markdown_content)
                    documents = []

                    for chunk in tqdm.tqdm(chunks, desc="Procesando fragmentos"):
                        user_prompt = f"""
                        <Document starts>

                        {markdown_content}

                        <Document ends>

                        <Chunk starts>

                        {chunk}

                        <Chunk ends>
                        """
                        system_message = self.llm.create_message(system_prompt, role="system", llm_client=self.llm.llm_client)
                        user_message = self.llm.create_message(user_prompt, role="user", llm_client=self.llm.llm_client)

                        llm_response = self.llm.complete([system_message, user_message])
                        chunk_context = llm_response.text_response
                        llm_cost = llm_response.cost

                        contextual_chunk = chunk_context + "\n\n" + chunk.page_content
                        embedding = self.embeddings_instance.embed_documents([contextual_chunk])[0]

                        document = {
                            "chunk_id": f"chunk_{hash(contextual_chunk)}",
                            "parent_id": f"parent_chunk_{document_id}",
                            "grand_parent_id": "",
                            "document_name": filename,
                            "chunk": contextual_chunk,
                            "chunk_metadata": str(chunk.metadata),
                            "text_vector": embedding,
                        }

                        documents.append(document)
                        total_indexing_cost += llm_cost

                    self.search_client.upload_documents(documents)
                    logger.info(f"{filename} indexado con {len(chunks)} fragmentos.")
                    logger.info(f"Coste total de indexación: {total_indexing_cost} euros.")

        except Exception as e:
            logger.error(f"Error durante la indexación contextual: {str(e)}")
