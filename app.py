import os
import streamlit as st
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Agente Clínica VidaSaúde", page_icon="🏥")
st.title("🏥 Assistente Virtual - Clínica VidaSaúde")

# Verificar se a API Key está configurada no ambiente
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("⚠️ Erro: Variável 'GOOGLE_API_KEY' não encontrada no arquivo .env!")
    st.info(
        "Crie um arquivo '.env' na raiz do projeto com o conteúdo: GOOGLE_API_KEY=sua_chave"
    )
    st.stop()


@st.cache_resource
def carregar_e_processar_pdf():
    # Carregar PDF
    loader = PyPDFLoader("documentos/manual_clinica.pdf")
    docs = loader.load()

    # Dividir em pedaços (chunks)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # Embeddings locais da HuggingFace (gratuitos, rápidos e sem erro 404)
    # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

    # Criar Vector Store em memória
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever()


retriever = carregar_e_processar_pdf()

# Configurar o LLM Gemini
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.3)

system_prompt = (
    "Você é um assistente virtual da Clínica VidaSaúde.\n"
    "Responda à pergunta do usuário utilizando APENAS o contexto fornecido abaixo.\n"
    "Se não souber a resposta com base no contexto, diga educadamente que não possui essa informação.\n\n"
    "Contexto:\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Pipeline RAG com LCEL
rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Interface de Chat no Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_question := st.chat_input("Como posso ajudar hoje?"):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        answer = rag_chain.invoke(user_question)
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
