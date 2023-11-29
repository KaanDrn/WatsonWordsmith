import os
import pathlib
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class TextLoader():
    def __init__(self, book_path:pathlib.Path):
        """
        Parameters
        ----------
        book_path:
            book as path with all chapters in it in the structure:
            "chapter_<chapter_number>__part_<part_number>"
        """
        self.book_path = book_path

        self.corpus = None
        self.splitter = None
        self.splits = None
        return

    def load_from_txt_file(self):
        self.corpus = DirectoryLoader(self.book_path).load()
        return self.corpus

    def set_splitter(self):
        self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=20,
                separators=["\n\n", "\n", " ", ""]
        )
        return

    def get_splits(self):
        self.splits = self.splitter.split_documents(self.corpus)
        return
