"""
Steps 4 & 5: Chained Prompts + RAG System
Implements a two-stage chain:
  Stage 1 (Condense) — rewrites the user's question using conversation history
                        into a standalone question.
  Stage 2 (Answer)   — retrieves relevant documents and generates a structured,
                        insight-rich response from the LLM.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def build_llm(model: str = "gpt-4o-mini", temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=temperature)


# ---------------------------------------------------------------------------
# Stage 1: Condense prompt
# Rewrites follow-up questions into self-contained standalone questions
# so that retrieval is context-independent.
# ---------------------------------------------------------------------------

CONDENSE_TEMPLATE = """Given the conversation history and a follow-up question, \
rewrite the follow-up question into a standalone question that contains all \
necessary context. If the question is already standalone, return it unchanged.

Conversation history:
{chat_history}

Follow-up question: {question}

Standalone question:"""

CONDENSE_PROMPT = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=CONDENSE_TEMPLATE,
)


def format_chat_history(messages: list[BaseMessage]) -> str:
    """Convert a list of BaseMessage objects into a readable string."""
    if not messages:
        return "None"
    lines = []
    for msg in messages:
        role = "Human" if msg.type == "human" else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage 2: Answer prompt
# Guides the LLM to produce structured, factual BI insights from context.
# ---------------------------------------------------------------------------

ANSWER_TEMPLATE = """You are InsightForge, an expert Business Intelligence Assistant \
specializing in sales data analysis. Use ONLY the data provided in the context below \
to answer the question. Do not invent numbers.

Guidelines:
- Lead with a direct answer to the question.
- Support your answer with specific figures from the context.
- Identify trends, comparisons, or noteworthy patterns where relevant.
- If the context does not contain enough information, say so clearly.
- Format your response with clear sections if the answer has multiple parts.
- Use dollar signs and commas when quoting sales figures.

Context:
{context}

Question: {question}

Answer:"""

ANSWER_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=ANSWER_TEMPLATE,
)


# ---------------------------------------------------------------------------
# Helper: format retrieved documents into a single string
# ---------------------------------------------------------------------------

def format_docs(docs) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# ---------------------------------------------------------------------------
# Chain builder
# ---------------------------------------------------------------------------

def build_rag_chain(retriever, llm: ChatOpenAI):
    """
    Returns a callable chain that accepts:
        {"question": str, "chat_history": list[BaseMessage]}
    and returns an answer string.
    """

    # Stage 1: condense follow-up question → standalone question
    condense_chain = (
        {
            "chat_history": RunnableLambda(
                lambda x: format_chat_history(x.get("chat_history", []))
            ),
            "question": RunnablePassthrough() | RunnableLambda(lambda x: x["question"]),
        }
        | CONDENSE_PROMPT
        | llm
        | StrOutputParser()
    )

    # Stage 2: retrieve → answer
    def retrieve_and_answer(inputs: dict) -> str:
        standalone_question = inputs["standalone_question"]
        docs = retriever.invoke(standalone_question)
        context = format_docs(docs)
        answer_input = {"context": context, "question": standalone_question}
        return (ANSWER_PROMPT | llm | StrOutputParser()).invoke(answer_input)

    full_chain = (
        RunnablePassthrough.assign(
            standalone_question=condense_chain
        )
        | RunnableLambda(retrieve_and_answer)
    )

    return full_chain


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from step3_retriever import build_retriever

    retriever = build_retriever(k=3)
    llm = build_llm()
    chain = build_rag_chain(retriever, llm)

    test_questions = [
        "What are the total sales by region?",
        "Which product had the highest average sales?",
        "How did sales trend year over year?",
        "Which customer age group spends the most?",
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        answer = chain.invoke({"question": q, "chat_history": []})
        print(f"A: {answer}")
        print("-" * 60)
