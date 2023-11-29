import os
import openai
import sys
from dotenv import load_dotenv, find_dotenv
import pathlib


from utils import document_handler as dh

load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

if __name__ == "__main__":
    book_path = pathlib.Path('data/books/animal_farm')
    loader_obj = dh.TextLoader(book_path)

    corpus = loader_obj.load_from_txt_file()
    loader_obj.set_splitter()
    loader_obj.get_splits()

    print('done')