from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Weaviate
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter

import os


loader = PyPDFLoader("vector.pdf")
document = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
docs = text_splitter.split_documents(document)
embeddings = OpenAIEmbeddings()
vectorstore = Weaviate.from_documents(docs, embeddings, by_text=False)

PROMPT="""Leveraging the capabilities of OpenAI, the Assistant emerges as a sophisticated language model meticulously trained through the assimilation of data embedded within the Vector Store.

The inherent design of the Assistant empowers it to furnish valuable aid in deciphering the wealth of information residing within the data repository. From addressing straightforward inquiries to unraveling intricate concepts and participating in discussions across a diverse array of subjects delineated in the associated paper, the Assistant thrives in facilitating seamless interactions. Functioning as a linguistic model, the Assistant orchestrates textual responses akin to human dialogue, thereby nurturing organic and contextually relevant conversations that seamlessly integrate into the ongoing discourse.

At its core, the Assistant transcends its role as a mere tool to evolve into a versatile companion, capable of catering to an extensive spectrum of tasks. It serves as a wellspring of insights and particulars, readily dispensing invaluable information spanning the breadth of topics encapsulated within the paper. Whether you seek elucidation on a specific query or aspire to delve into an engrossing dialogue on a particular theme, the Assistant is primed to meet your needs.

In instances where the Assistant encounters uncharted territory, its response gracefully conveys, 'The information you're seeking resides beyond the scope of my current knowledge.'
{context}
User: {question}
AI:"""


prompt_template = PromptTemplate(
    template=PROMPT,
    input_variables=["context", "question"]
)

llm = OpenAI(temperature=0.5)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
vectorstore = vectorstore.as_retriever()
combine_docs_chain_kwargs = {'prompt': prompt_template}
question_answer = ConversationalRetrievalChain.from_llm(llm, vectorstore, memory=memory,
                                                        combine_docs_chain_kwargs=combine_docs_chain_kwargs,)
