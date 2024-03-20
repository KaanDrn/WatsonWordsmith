import os
import pathlib

from dotenv import load_dotenv, find_dotenv
from langchain_community.llms import HuggingFaceHub

load_dotenv(find_dotenv())

from utils.database_utils import VectorDatabase, Retriever
from utils import chat_utils

if __name__ == "__main__":
    # define which book you want to chat with
    book_path = pathlib.Path('data/books/animal_farm')

    # create vector_db object and determine if a vector db already exists or
    # you want to create a new one
    vector_db_obj = VectorDatabase(book_path)
    if os.path.exists(book_path.joinpath('database')):
        vector_db_obj.set_vector_store()
    else:
        splits = VectorDatabase.get_all_documents_for_book(book_path)
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
    question = "What animal is Moses?"
    # question = "What animal is Napoleon?"
    # question = "What is the name of the farm, the animals live in?"
    # question = "What was the name of the farm before that?"
    # question = "Why did Snowball leave the farm?"

    answer = answer_obj.answer(question)

    print(answer)
