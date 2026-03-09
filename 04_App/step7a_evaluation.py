"""
Step 7a: Model Evaluation
Scores the RAG assistant's responses against ground-truth Q&A pairs
using an LLM-as-judge grading prompt (equivalent to QAEvalChain).
"""

import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from step6_assistant import InsightForgeAssistant

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ---------------------------------------------------------------------------
# LLM-as-judge grader (replaces QAEvalChain)
# ---------------------------------------------------------------------------

GRADE_PROMPT = PromptTemplate(
    input_variables=["query", "answer", "result"],
    template=(
        "You are a grader evaluating whether a model's answer is correct.\n\n"
        "Question: {query}\n"
        "Expected answer: {answer}\n"
        "Model answer: {result}\n\n"
        "Is the model answer correct? Reply with exactly one word: CORRECT or INCORRECT."
    ),
)


def grade_response(llm: ChatOpenAI, query: str, expected: str, predicted: str) -> str:
    chain = GRADE_PROMPT | llm | StrOutputParser()
    verdict = chain.invoke({"query": query, "answer": expected, "result": predicted})
    return verdict.strip().upper()

# ---------------------------------------------------------------------------
# Ground-truth Q&A pairs (answers derived from the actual dataset)
# ---------------------------------------------------------------------------

QA_PAIRS = [
    {
        "query": "What are the four products sold in this dataset?",
        "answer": "The four products are Widget A, Widget B, Widget C, and Widget D.",
    },
    {
        "query": "How many regions are covered in the sales data?",
        "answer": "There are four regions: North, South, East, and West.",
    },
    {
        "query": "What is the date range of the sales data?",
        "answer": "The sales data spans from January 2022 to December 2024.",
    },
    {
        "query": "What are the customer genders recorded in the dataset?",
        "answer": "The dataset records Male and Female customers.",
    },
    {
        "query": "What is the scale used for customer satisfaction scores?",
        "answer": "Customer satisfaction is measured on a scale from 1.0 to 5.0.",
    },
    {
        "query": "What customer age range is represented in the data?",
        "answer": "Customer ages range from 19 to 69 years old.",
    },
    {
        "query": "How many years of sales data are available?",
        "answer": "Three years of sales data are available: 2022, 2023, and 2024.",
    },
    {
        "query": "What types of analysis does InsightForge support?",
        "answer": (
            "InsightForge supports sales performance analysis by time period, "
            "product and regional analysis, customer segmentation by demographics, "
            "and customer satisfaction analysis."
        ),
    },
]


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(assistant: InsightForgeAssistant) -> list[dict]:
    """
    Generate predictions for each QA pair and grade them with an LLM judge.
    Returns a list of result dicts with query, answer, prediction, and grade.
    """
    eval_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    results = []

    for i, pair in enumerate(QA_PAIRS):
        assistant.reset()  # fresh context per question
        print(f"  Q: {pair['query'][:70]}")
        prediction = assistant.ask(pair["query"])
        print(f"  A: {prediction[:120]}\n")

        grade = grade_response(eval_llm, pair["query"], pair["answer"], prediction)

        results.append({
            "index": i + 1,
            "query": pair["query"],
            "expected": pair["answer"],
            "predicted": prediction,
            "grade": grade,
        })

    return results


def print_report(results: list[dict]) -> None:
    correct = sum(1 for r in results if "CORRECT" in r["grade"].upper())
    total = len(results)
    accuracy = correct / total * 100

    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)
    print(f"Score: {correct}/{total} correct  ({accuracy:.1f}%)\n")

    for r in results:
        status = "PASS" if "CORRECT" in r["grade"].upper() else "FAIL"
        print(f"[{status}] Q{r['index']}: {r['query']}")
        if status == "FAIL":
            print(f"       Expected : {r['expected'][:100]}")
            print(f"       Predicted: {r['predicted'][:100]}")
        print()

    print("=" * 70)
    return accuracy


def save_report(results: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Report saved to: {path}")


if __name__ == "__main__":
    assistant = InsightForgeAssistant()
    results = run_evaluation(assistant)
    print_report(results)

    report_path = os.path.join(
        os.path.dirname(__file__), "..", "05_Evaluation", "eval_report.json"
    )
    save_report(results, report_path)
