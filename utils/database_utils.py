import os
import pathlib
import shutil

from langchain.chains.query_constructor.schema import AttributeInfo
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import SelfQueryRetriever, \
    ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.vectorstores import Chroma

from utils import document_handler as dh


class VectorDatabase:
    def __init__(self,
                 book_path: pathlib.Path,
                 embeddings=HuggingFaceEmbeddings()):
        """
        Parameters
        ----------
        book_path:
            path to the vector database, should be the same as the book path
        """
        self.vector_store_dir = str(book_path.joinpath('database'))
        self.embeddings = embeddings

        self.vector_db = None

    def create_vector_store(self, splits, delete_existing_db=True):
        if delete_existing_db and os.path.isdir(self.vector_store_dir):
            shutil.rmtree(self.vector_store_dir)

        self.vector_db = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=self.vector_store_dir
        )
        self.vector_db.persist()

    def set_vector_store(self):
        self.vector_db = Chroma(
                persist_directory=self.vector_store_dir,
                embedding_function=self.embeddings
        )
        # ToDo: add exception, if DB does not exist
        return

    def set_llm(self, llm):
        self.llm = llm

    def set_meta_data_fields(self):
        query_metadata = [
            AttributeInfo(
                    name="source",
                    description="The raw data, which contains the book.",
                    type="string",
            ),
            AttributeInfo(
                    name="chapter",
                    description="The chapter from the book",
                    type="integer",
            ),
            AttributeInfo(
                    name="subchapter",
                    description="The subchapter is part of the chapter and "
                                "therefore hierarchically lower.",
                    type="integer",
            ),
        ]
        self.query_metadata = query_metadata


class Retriever:
    def __init__(self, llm, vector_db):
        self.llm = llm
        self.vector_db = vector_db
        self.retriever = None

    def set_retriever(self, retriever_type='self', query_metadata=None):
        if retriever_type == 'self':
            document_content_description = "Book"
            retriever = SelfQueryRetriever.from_llm(
                    self.llm,
                    self.vector_db,
                    document_content_description,
                    query_metadata,
                    verbose=True
            )
        elif retriever_type == 'compress':
            compressor = LLMChainExtractor.from_llm(self.llm)
            retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=self.vector_db.as_retriever()
            )
        elif retriever_type == 'vector_db':
            retriever = self.vector_db.as_retriever(
                    search_type='mmr',
                    search_kwargs={"k": 6,
                                   "fetch_k": 18})
        self.retriever = retriever


def get_all_documents_for_book(book_path):
    """
    gets all documents for book and returns the already splitted data
    Parameters
    ----------
    book_path

    Returns
    -------

    """
    loader_obj = dh.TextLoader(book_path)
    corpus = loader_obj.load_from_txt_file()
    loader_obj.set_splitter()
    splits = loader_obj.get_splits()
    return splits
