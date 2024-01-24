import os
import pathlib
from datetime import datetime

import streamlit as st

st.title(":left_speech_bubble::male-detective: Watson Wordsmith:")

if "chats" not in st.session_state:
    st.session_state.chats = {}


def create_book_selector():
    global selected_book
    global default_text
    default_text = "Select a book you want to talk about"
    BOOKS = [ibook.replace('_', ' ').title() for ibook in
             os.listdir(pathlib.Path('data/books'))]
    st.sidebar.subheader("Chats")
    selected_book = st.sidebar.selectbox("",
                                         [default_text] + BOOKS)


create_book_selector()

# Button to create a new chat with the selected book
if st.sidebar.button("Create New Chat") and selected_book:
    if selected_book is not default_text:
        selected_chat = selected_book

        st.session_state.chats[selected_chat] = []

# Display messages for the selected chat
st.sidebar.subheader("Current Chat")

# Dropdown menu for selecting and deleting a chat
selected_chat = st.sidebar.selectbox("Select Chat",
                                     list(st.session_state.chats.keys()),
                                     index=0)
delete_chat_button = st.sidebar.button("Delete Current Chat")

# Delete the selected chat if the delete button is pressed
if delete_chat_button and selected_chat in st.session_state.chats:
    del st.session_state.chats[selected_chat]

if selected_chat:
    st.subheader(f"Current Chat: {selected_chat}")

    # Display previously entered messages for the selected chat
    for message in st.session_state.chats.get(selected_chat, []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input through a chat input field
    if prompt := st.chat_input("What is up?"):
        # Add user input to the selected chat in the session state
        st.session_state.chats[selected_chat].append(
                {"role": "user", "content": prompt})

        # Display user input
        with st.chat_message("user"):
            st.markdown(prompt)

        # Simulate assistant response (Placeholder for actual chatbot
        # interaction)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            full_response = "beep boop + %s" % date_time
            # Omitted chatbot interaction
            message_placeholder.markdown(full_response)

        # Add assistant response to the selected chat in the session state
        st.session_state.chats[selected_chat].append(
                {"role": "assistant", "content": full_response})
