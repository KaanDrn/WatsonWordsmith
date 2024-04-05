__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import pathlib
from datetime import datetime

import streamlit as st
from langchain_community.llms.huggingface_hub import HuggingFaceHub

from utils import chat_utils
from utils.database_utils import VectorDatabase, Retriever

st.title(":left_speech_bubble::male-detective: Watson Wordsmith:")
st.subheader("Your BookBuddy-Chatbot :books::open_book:")
if "initialized" not in st.session_state:
    st.session_state.initialized = False

BOOKS = ["1984 - George Orwell",
         "Animal Farm - George Orwell"]
NOW = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")


def init_states():
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = \
        st.secrets['HUGGINGFACEHUB_API_TOKEN']
    st.session_state.initialized = True
    st.session_state.create_button_visible = True
    st.session_state.add_new_book_visibile = False

    st.session_state.sidebar = st.sidebar
    st.session_state.container = st.container

    # chats
    st.session_state.current_chat = []
    st.session_state.all_chats = {}

    # create llm
    st.session_state.local_llm = HuggingFaceHub(
            repo_id="google/flan-t5-xxl",
            model_kwargs={
                #"max_new_tokens": 250,
                "temperature": 0.01
            },
    )


if not st.session_state.initialized:
    init_states()


def button_add_new_book():
    st.session_state.create_button_visible = False
    st.session_state.add_new_book_visibile = True


def create_new_chat():
    st.session_state.create_button_visible = True
    st.session_state.add_new_book_visibile = False

    chat_key = "%s\n%s" % (st.session_state.new_chat, NOW)

    st.session_state.current_chat = chat_key
    st.session_state.current_book_name = \
        get_book_name(st.session_state.new_chat)

    st.session_state.all_chats[chat_key] = \
        [
            {'role': 'ai',
             'message': f"Hi, I am Watson, lets talk about "
                        f"{st.session_state.current_book_name}."}
        ]

    # initialize vector db
    book_path = pathlib.Path(
            f'data/books/'
            f'{st.session_state.current_book_name.lower().replace(" ", "_")}')
    # create vector_db object and determine if a vector db already exists or
    # you want to create a new one
    vector_db_obj = VectorDatabase(book_path)
    if os.path.exists(book_path.joinpath('database')):
        vector_db_obj.set_vector_store()
    else:
        splits = vector_db_obj.get_all_documents_for_book()
        vector_db_obj.create_vector_store(splits)

    vector_db_obj.set_meta_data_fields()

    retriever_obj = Retriever(llm=st.session_state.local_llm,
                              vector_db=vector_db_obj.vector_db)
    retriever_obj.set_retriever(retriever_type='vector_db',
                                query_metadata=vector_db_obj.query_metadata)

    st.session_state.answer_obj = chat_utils.AnswerMe(
            st.session_state.local_llm,
            vector_db_obj.vector_db,
            retriever_obj.retriever)


def delete_chat(**kwargs):
    del st.session_state.all_chats[kwargs.get('selected_chat')]
    try:
        st.session_state.current_chat = \
            list(st.session_state.all_chats.keys())[-1]
    except IndexError:
        st.session_state.current_chat = []


def select_chat(**kwargs):
    st.session_state.current_chat = kwargs.get('selected_chat')
    st.session_state.current_book_name = get_book_name(
            kwargs.get('selected_chat'))
    st.session_state.current_chat_number = kwargs.get('selected_chat_number')


def answer_me(prompt):
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = st.session_state.answer_obj.answer(prompt)
    st.session_state.all_chats[st.session_state.current_chat].append(
            {'role': 'ai',
             'message': answer
             # 'message': f"question: {prompt} answer: Beep Boop this is an "
             #            f"answer" + " - " + NOW
             }
    )
    return


# building the website from here on
# manages the sidebar
with st.session_state.sidebar:
    # if you want to add a new chat
    if st.session_state.create_button_visible:
        st.button("Create new chat",
                  key='create_new_chat_button',
                  on_click=button_add_new_book,
                  disabled=False
                  )
    if st.session_state.add_new_book_visibile:
        selection = st.session_state.sidebar.selectbox(label="Available books",
                                                       key='book_selection',
                                                       options=BOOKS)
        st.session_state.sidebar.button("Start chat",
                                        key='start_chat_button',
                                        on_click=create_new_chat)
        st.session_state.new_chat = selection

    # if you have selected a book, this text will appear
    if st.session_state.current_chat:
        st.session_state.col_chat_name, \
        st.session_state.col_delete_chat, \
        st.session_state.col_select_chat = \
            st.session_state.sidebar.columns([0.7, 0.15, 0.15])

        for inum_chat, ichat in enumerate(st.session_state.all_chats):
            with st.session_state.container(border=True):
                st.session_state.col_chat_name.text(ichat)
                st.session_state.col_delete_chat.button(
                        ":wastebasket:",
                        on_click=delete_chat,
                        key=f'del_button_{inum_chat}',
                        kwargs={'selected_chat': ichat})
                st.session_state.col_select_chat.button(
                        ":white_check_mark:",
                        on_click=select_chat,
                        key=f'sel_button_{inum_chat}',
                        kwargs={'selected_chat': ichat,
                                'selected_chat_number': inum_chat})


def get_book_name(current_chat):
    return current_chat.split(' - ')[0]


# manages the chat window
if st.session_state.current_chat:
    prompt = st.chat_input('Talk to me...')

    for imessage in st.session_state.all_chats[st.session_state.current_chat]:
        with st.chat_message(imessage.get('role')):
            st.write(imessage.get('message'))

    if prompt:
        st.session_state.all_chats[st.session_state.current_chat].append(
                {'role': 'user',
                 'message': prompt}
        )
        answer = answer_me(prompt)
        st.rerun()
