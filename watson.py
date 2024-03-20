import os
import pathlib

import openai
from dotenv import load_dotenv, find_dotenv
from langchain import LLMChain
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate

load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

from utils import document_handler as dh
from utils.database_utils import VectorDatabase, Retriever
from utils import chat_utils
from transformers import pipeline, AutoTokenizer


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


if __name__ == "__main__":
    # define which book you want to chat with
    book_path = pathlib.Path('data/books/animal_farm')

    # create vector_db object and determine if a vector db already exists or
    # you want to create a new one
    vector_db_obj = VectorDatabase(book_path)
    if os.path.exists(book_path.joinpath('database')):
        vector_db_obj.set_vector_store()
    else:
        splits = get_all_documents_for_book(book_path)
        vector_db_obj.create_vector_store(splits)

    vector_db_obj.set_meta_data_fields()
    # create llm
    local_llm = HuggingFaceHub(
            repo_id="google/flan-t5-large",
            model_kwargs={
                # "max_new_tokens": 200,
                "temperature": 0.1
            },
    )

    retriever_obj = Retriever(llm=local_llm,
                              vector_db=vector_db_obj.vector_db)
    retriever_obj.set_retriever(retriever_type='vector_db',
                                query_metadata=vector_db_obj.query_metadata)

    answer_obj = chat_utils.AnswerMe(local_llm,
                                     vector_db_obj.vector_db,
                                     retriever_obj.retriever)

    # sample questions, that can be asked
    # question = "What animal is Napoleon?"
    question = "What animal is Moses?"
    # question = "What is the name of the farm, the animals live in?"
    # question = "what was the name of the farm before it became animal farm?"
    # question = "Why does snowball no longer live on the farm?"

    answer = answer_obj.answer(question)

    print(answer)
