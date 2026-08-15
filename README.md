# 🏥 Agente Virtual - Clínica VidaSaúde (RAG)

Este projeto é um Agente Inteligente capaz de responder a dúvidas de pacientes com base na documentação oficial da **Clínica VidaSaúde** utilizando técnicas de RAG (Retrieval-Augmented Generation).

🔗 **URL da Aplicação:** []

---

## 📐 Arquitetura da Solução
1. **Documento (PDF):** Manual de processos e informações da clínica.
2. **Processamento (LangChain):** Leitura, divisão em chunks e criação de embeddings com ChromaDB.
3. **LLM (Google Gemini 3.5 Flash Lite):** Geração de respostas baseadas estritamente no contexto recuperado.
4. **Interface (Streamlit):** Web app interativa e responsiva.

---

## 🛠️ Tecnologias Utilizadas
- **Linguagem:** Python 3.10+
- **Framework de IA:** LangChain
- **LLM e Embeddings:** Google Gemini AI
- **Banco Vetorial:** ChromaDB
- **Interface & Deploy:** Streamlit & Streamlit Cloud

---

## 🚀 Como Executar Localmente
```bash
# 1. Clonar o repositório
git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Executar o Streamlit
streamlit run app.py
```

---

## 💬 Exemplos de Uso

Pergunta: Quais são os horários de coleta para exame de sangue?

Resposta: A coleta para exames de sangue é realizada de segunda a sexta-feira, das 07h às 10h, por ordem de chegada.

Pergunta: Vocês aceitam o convênio Unimed?

Resposta: Sim, a clínica atende Unimed, Bradesco Saúde e consultas particulares.
