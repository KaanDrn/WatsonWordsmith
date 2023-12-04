import os
import pathlib
import shutil

from langchain.chains.query_constructor.base import AttributeInfo
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo


class TextLoader():
    def __init__(self, book_path: pathlib.Path):
        """
        Parameters
        ----------
        book_path:
            book as path with all chapters in it in the structure:
            "chapter_<chapter_number>__part_<part_number>"
        """
        self.book_path = book_path.joinpath('content')
        self.vector_store_dir = self.book_path

        self.book_files = os.listdir(self.book_path)

        self.corpus = None
        self.splitter = None
        self.splits = None
        return

    def load_from_txt_file(self):
        self.corpus = DirectoryLoader(self.book_path).load()
        self.add_chapter_meta_data()
        return self.corpus

    def add_chapter_meta_data(self):
        # ToDo: use openlibrary.org to add relevant information about the
        #  books to meta data
        for icorpus in self.corpus:
            file_name = icorpus.metadata.get('source')
            file_name = file_name.split('\\')[-1].split('.')[0]

            temp = file_name.split('__')
            chapter = int(temp[0].split('_')[1])
            subchapter = int(temp[1].split('_')[1].split('.')[0])

            icorpus.metadata['chapter'] = chapter
            icorpus.metadata['subchapter'] = subchapter

    def set_splitter(self):
        self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", " ", ""]
        )
        return

    def get_splits(self):
        self.splits = self.splitter.split_documents(self.corpus)
        return self.splits


class VectorDatabase:
    def __init__(self,
                 vector_db_path: pathlib.Path,
                 embeddings=OpenAIEmbeddings()):
        """
        Parameters
        ----------
        vector_db_path:
            path to the vector database, should be the same as the book path
        """
        self.vector_store_dir = str(vector_db_path.joinpath('database'))
        self.embeddings = embeddings

        self.splits = None
        self.vector_db = self.set_vector_store()

        self.llm = self.set_llm()
        self.query_metadata = self.set_meta_data_fields()
        self.retriever = self.set_retriever()

    def create_vector_store(self, splits,
                            delete_existing_db=True):
        if delete_existing_db and os.path.isdir(self.vector_store_dir):
            shutil.rmtree(self.vector_store_dir)

        self.vector_db = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=self.vector_store_dir
        )
        self.vector_db.persist()

    def set_vector_store(self):
        vector_db = Chroma(
                persist_directory=self.vector_store_dir,
                embedding_function=self.embeddings
        )
        return vector_db

    def get_relevant_passages(self, question):
        return self.retriever.get_relevant_documents(question)

    @staticmethod
    def set_llm():
        return OpenAI(temperature=0)

    @staticmethod
    def set_meta_data_fields():
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
        return query_metadata

    def set_retriever(self):
        document_content_description = "Book"
        retriever = SelfQueryRetriever.from_llm(
                self.llm,
                self.vector_db,
                document_content_description,
                self.query_metadata,
                verbose=True
        )
        return retriever
