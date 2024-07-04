import os
import pathlib

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class TextLoader:
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
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", " ", ""]
        )
        return

    def get_splits(self):
        self.splits = self.splitter.split_documents(self.corpus)
        return self.splits
