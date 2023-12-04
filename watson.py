import os
import openai
import sys
from dotenv import load_dotenv, find_dotenv
import pathlib

load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

from utils import document_handler as dh
from utils.document_handler import VectorDatabase


if __name__ == "__main__":
    book_path = pathlib.Path('data/books/animal_farm')
    loader_obj = dh.TextLoader(book_path)

    corpus = loader_obj.load_from_txt_file()
    loader_obj.set_splitter()
    splits = loader_obj.get_splits()

    vector_db_obj = VectorDatabase(book_path)
    vector_db_obj.create_vectore_store(splits)

    question = "What animal is Napoleon?"
    question = "What is the name of the farm, the animals live in?"

    vector_db_obj.get_relevant_passages(question)

    print('done')
