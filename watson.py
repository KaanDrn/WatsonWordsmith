import os
import pathlib

import openai
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI

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

    answer_obj = dh.AnswerMe(ChatOpenAI(model_name="gpt-3.5-turbo",
                                        temperature=0.1,
                                        max_tokens=500),
                             vector_db_obj.vector_db,
                             vector_db_obj.retriever)

    # sample questions, that can be asked
    question = "What animal is Napoleon?"
    question = "What is the name of the farm, the animals live in? Only Use " \
               "data from the first chapter"
    question = "Answer like Donald Trump. What animal is Moses?"
    question = "What does Napoleon represent in real life?"
    question = "What does Boxer stand for in the " \
               "book?"

    answer = answer_obj.answer(question)

    print('done')
