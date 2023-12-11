import os
import openai
import sys
from dotenv import load_dotenv, find_dotenv
import pathlib

from langchain.embeddings import HuggingFaceEmbeddings

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



    answer_obj = loader_obj.AnswerMe(vector_db_obj.llm,
                                     vector_db_obj.vector_db,
                                     vector_db_obj.retriever)

    #docs = vector_db_obj.answer(question)

    # question = "What animal is Napoleon?"
    question = "What is the name of the farm, the animals live in? Only Use " \
               "data from the first chapter"
    question = "What animal is Moses?"

    print('done')
