"""
Step 2: Knowledge Base Creation
Converts computed summary statistics into text documents and indexes
them in a FAISS vector store for retrieval by the RAG system.
"""

import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from step1_data_preparation import load_data, compute_summary_statistics

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

VECTORSTORE_PATH = os.path.join(os.path.dirname(__file__), "..", "05_VectorStore")


def build_documents(stats: dict) -> list[Document]:
    """Convert summary statistics into LangChain Documents."""
    docs = []

    # --- Overall sales ---
    o = stats["overall"]
    docs.append(Document(
        page_content=(
            f"Overall Sales Summary: Total sales are ${o['total_sales']:,}. "
            f"The average daily sale is ${o['mean_sales']:,} with a median of ${o['median_sales']:,} "
            f"and a standard deviation of ${o['std_sales']:,}. "
            f"Sales range from ${o['min_sales']:,} to ${o['max_sales']:,}."
        ),
        metadata={"category": "overall", "topic": "sales_summary"}
    ))

    # --- Sales by year ---
    year_lines = []
    for year, v in stats["by_year"].items():
        year_lines.append(
            f"{year}: total=${v['total']:,.0f}, average=${v['mean']:,.2f}, median=${v['median']:,.2f}"
        )
    docs.append(Document(
        page_content="Annual Sales Performance:\n" + "\n".join(year_lines),
        metadata={"category": "time", "topic": "annual_sales"}
    ))

    # --- Sales by quarter ---
    quarter_lines = []
    for q in stats["by_quarter"]:
        quarter_lines.append(f"Q{q['Quarter']} {q['Year']}: total=${q['total']:,.0f}")
    docs.append(Document(
        page_content="Quarterly Sales Performance:\n" + "\n".join(quarter_lines),
        metadata={"category": "time", "topic": "quarterly_sales"}
    ))

    # --- Sales by month ---
    month_lines = []
    for month, v in stats["by_month"].items():
        month_lines.append(f"{month}: total=${v['total']:,.0f}, average=${v['mean']:,.2f}")
    docs.append(Document(
        page_content="Monthly Sales Performance (aggregated across all years):\n" + "\n".join(month_lines),
        metadata={"category": "time", "topic": "monthly_sales"}
    ))

    # --- Sales by product ---
    product_lines = []
    for product, v in stats["by_product"].items():
        product_lines.append(
            f"{product}: total=${v['total']:,.0f}, average=${v['mean']:,.2f}, "
            f"median=${v['median']:,.2f}, transactions={v['count']}"
        )
    docs.append(Document(
        page_content="Product Sales Performance:\n" + "\n".join(product_lines),
        metadata={"category": "product", "topic": "product_sales"}
    ))

    # --- Sales by region ---
    region_lines = []
    for region, v in stats["by_region"].items():
        region_lines.append(
            f"{region}: total=${v['total']:,.0f}, average=${v['mean']:,.2f}, "
            f"median=${v['median']:,.2f}, transactions={v['count']}"
        )
    docs.append(Document(
        page_content="Regional Sales Performance:\n" + "\n".join(region_lines),
        metadata={"category": "region", "topic": "regional_sales"}
    ))

    # --- Product x Region breakdown ---
    pr_lines = []
    for row in stats["product_region"]:
        pr_lines.append(f"{row['Product']} in {row['Region']}: total=${row['total']:,.0f}")
    docs.append(Document(
        page_content="Product Performance by Region:\n" + "\n".join(pr_lines),
        metadata={"category": "product_region", "topic": "product_region_breakdown"}
    ))

    # --- Customer demographics ---
    d = stats["demographics"]
    gender_dist = ", ".join(f"{k}: {v}" for k, v in d["gender_distribution"].items())
    age_group_lines = "\n".join(
        f"  Age {group}: total=${v['total']:,.0f}, average=${v['mean']:,.2f}"
        for group, v in d["sales_by_age_group"].items()
    )
    gender_sales_lines = "\n".join(
        f"  {gender}: total=${v['total']:,.0f}, average=${v['mean']:,.2f}"
        for gender, v in d["sales_by_gender"].items()
    )
    docs.append(Document(
        page_content=(
            f"Customer Demographics:\n"
            f"Average customer age is {d['age_mean']} years (std={d['age_std']}).\n"
            f"Gender distribution: {gender_dist}.\n"
            f"Sales by age group:\n{age_group_lines}\n"
            f"Sales by gender:\n{gender_sales_lines}"
        ),
        metadata={"category": "demographics", "topic": "customer_segmentation"}
    ))

    # --- Customer satisfaction ---
    s = stats["satisfaction"]
    sat_product_lines = "\n".join(f"  {p}: {score}/5.0" for p, score in s["by_product"].items())
    sat_region_lines = "\n".join(f"  {r}: {score}/5.0" for r, score in s["by_region"].items())
    docs.append(Document(
        page_content=(
            f"Customer Satisfaction:\n"
            f"Overall mean satisfaction: {s['overall_mean']}/5.0 "
            f"(median={s['overall_median']}, std={s['overall_std']}).\n"
            f"Satisfaction by product:\n{sat_product_lines}\n"
            f"Satisfaction by region:\n{sat_region_lines}"
        ),
        metadata={"category": "satisfaction", "topic": "customer_satisfaction"}
    ))

    return docs


def create_vectorstore(docs: list[Document]) -> FAISS:
    """Embed documents and store in FAISS."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(docs, embeddings)
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"Vector store saved to: {VECTORSTORE_PATH}")
    return vectorstore


def load_vectorstore() -> FAISS:
    """Load an existing FAISS vector store from disk."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)


if __name__ == "__main__":
    print("Loading data...")
    df = load_data()

    print("Computing statistics...")
    stats = compute_summary_statistics(df)

    print("Building documents...")
    docs = build_documents(stats)
    print(f"  Created {len(docs)} documents.")
    for doc in docs:
        print(f"  - [{doc.metadata['topic']}] {doc.page_content[:80]}...")

    print("\nCreating and saving vector store...")
    vectorstore = create_vectorstore(docs)

    print("\nVerifying retrieval...")
    results = vectorstore.similarity_search("What are the total sales by region?", k=2)
    for i, r in enumerate(results, 1):
        print(f"\n  Result {i} [{r.metadata['topic']}]:\n  {r.page_content[:200]}")

    print("\nKnowledge base creation complete.")
