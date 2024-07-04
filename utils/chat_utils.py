from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import \
    CONDENSE_QUESTION_PROMPT
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms.ollama import Ollama


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
        return answer['answer'].split("Answer:")[-1]

    def set_template(self):
        self.template = \
            """
            Your name is watson and you are a friedly but professional 
            chatbot, that answers questions from a book, with the following 
            context: {context}
            Answer the following question with the context, do not make up 
            content, that is not mentioned in the context:
            Question: {question}
            You answer should be simple and should sound like a 
            professional literature expert. Help the user to improve his 
            question if you can not find the answer in the context.
            Helpful Answer:"""

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
                #return_source_documents=True,
                memory=self.memory,
                chain_type='stuff',
                combine_docs_chain_kwargs={'prompt': prompt_template},
                verbose=False,
        )

def create_local_llm(model_name='llama3:8b'):
    return Ollama(model=model_name,
                  temperature=0.1)
