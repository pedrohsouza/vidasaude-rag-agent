import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(
    page_title="Agente Clínica VidaSaúde", page_icon="🏥", layout="centered"
)
st.title("🏥 Assistente Virtual - Clínica VidaSaúde")
st.caption("Tire suas dúvidas sobre consultas, exames e convênios da clínica.")

# 1. Validação da API Key
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("⚠️ Chave 'GOOGLE_API_KEY' não encontrada no arquivo .env!")
    st.stop()


# 2. Processamento do PDF e RAG
@st.cache_resource
def carregar_e_processar_pdf():
    loader = PyPDFLoader("documentos/manual_clinica.pdf")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever()


retriever = carregar_e_processar_pdf()

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


rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 3. Inicialização do Histórico e Mensagem de Boas-Vindas
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Olá! 👋 Sou o assistente virtual da **Clínica VidaSaúde**. Como posso ajudar você hoje?",
        }
    ]

# 4. Renderização do Histórico do Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Seção de Perguntas Frequentes (Pills)
st.markdown("##### 💡 Sugestões de perguntas frequentes:")
selected_pill = st.pills(
    "Selecione uma dúvida comum:",
    options=[
        "Quais são os horários de coleta para exame de sangue?",
        "Vocês aceitam o convênio Unimed?",
    ],
    label_visibility="collapsed",
)

# 6. Captura de Entrada (seja digitada ou clicada nas pills)
user_prompt = None

# Input de texto do usuário
if text_input := st.chat_input("Digite sua dúvida aqui..."):
    user_prompt = text_input
elif selected_pill:
    # Evita reexecução contínua da mesma pill clicada
    if (
        "last_pill" not in st.session_state
        or st.session_state.last_pill != selected_pill
    ):
        user_prompt = selected_pill
        st.session_state.last_pill = selected_pill

# 7. Execução da Resposta com o Agente
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando manual da clínica..."):
            answer = rag_chain.invoke(user_prompt)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
