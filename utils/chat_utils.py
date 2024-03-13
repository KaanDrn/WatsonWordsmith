from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


class AnswerMe:
    def __init__(self, llm, vector_db, retriever):
        self.template = None
        self.answerer = None
        self.llm = llm
        self.vector_db = vector_db
        self.retriever = retriever

        self.set_template()
        self.set_memory()

        self.set_conversation_answerer()
        # self.set_answerer()

    def get_relevant_documents(self, question):
        return self.retriever.get_relevant_documents(query=question)

    def set_answerer(self):
        """
        Does not have context capacity, therefore will not be used now
        Returns
        -------

        """
        prompt_template = PromptTemplate.from_template(self.template)
        self.answerer = RetrievalQA.from_chain_type(
                self.llm,
                retriever=self.retriever,
                # return_source_documents=True,
                chain_type_kwargs={'prompt': prompt_template},
                # chain_type='stuff'
        )

    def answer(self, question):
        answer = self.answerer({'question': question})
        return answer['answer']

    def set_template(self):
        self.template = \
            """
            Answer like a literature expert, who someone asked for their 
            opinion on a certain part of a book and give a medium long 
            answer to each question. 
            Mention in which chapter or 
            subchapter the answer can be looked up.
            
            {context}
            Question: {question}
            Answer:"""

    def set_memory(self):
        self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
        )

    def set_conversation_answerer(self):
        prompt_template = PromptTemplate.from_template(self.template)
        self.answerer = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                memory=self.memory,
                combine_docs_chain_kwargs={'prompt': prompt_template}
        )
