"""
Step 6: Memory Integration
Wraps the RAG chain with conversation memory so the assistant remembers
previous turns within a session and produces contextually relevant follow-up answers.

This module is the main entry point for the AI assistant logic.
"""

import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from step3_retriever import build_retriever
from step4_5_rag_chain import build_llm, build_rag_chain

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


class InsightForgeAssistant:
    """
    Stateful conversational BI assistant.

    Usage:
        assistant = InsightForgeAssistant()
        response = assistant.ask("What are total sales by region?")
        response = assistant.ask("Which of those performed best last year?")
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.2, k: int = 3):
        print("Initialising InsightForge assistant...")
        self.retriever = build_retriever(k=k)
        self.llm = build_llm(model=model, temperature=temperature)
        self.chain = build_rag_chain(self.retriever, self.llm)
        self._history: list[HumanMessage | AIMessage] = []
        print("Ready.\n")

    def ask(self, question: str) -> str:
        """Submit a question and get an answer. Conversation history is preserved."""
        answer = self.chain.invoke({
            "question": question,
            "chat_history": self._history,
        })

        self._history.append(HumanMessage(content=question))
        self._history.append(AIMessage(content=answer))

        return answer

    def reset(self) -> None:
        """Clear conversation history."""
        self._history.clear()
        print("Conversation history cleared.")

    def get_history(self) -> list[dict]:
        """Return conversation history as a list of role/content dicts."""
        result = []
        for msg in self._history:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            result.append({"role": role, "content": msg.content})
        return result


# ---------------------------------------------------------------------------
# Interactive CLI for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    assistant = InsightForgeAssistant()

    print("InsightForge — Business Intelligence Assistant")
    print("Type 'quit' to exit, 'reset' to clear history, 'history' to view chat.\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            assistant.reset()
            continue
        if user_input.lower() == "history":
            for turn in assistant.get_history():
                print(f"  [{turn['role'].upper()}] {turn['content'][:120]}")
            continue

        answer = assistant.ask(user_input)
        print(f"\nInsightForge: {answer}\n")
