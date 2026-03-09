"""
Step 3 (Part A): Custom Pandas Retriever
Extracts relevant statistics from the DataFrame at query time,
then combines results with the FAISS vector store for a hybrid retrieval.
"""

import os
import pandas as pd
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from pydantic import Field
from typing import List, Optional
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from step1_data_preparation import load_data, compute_summary_statistics

VECTORSTORE_PATH = os.path.join(os.path.dirname(__file__), "..", "05_VectorStore")


class PandasStatsRetriever(BaseRetriever):
    """
    Hybrid retriever that:
      1. Detects keywords in the query to pull precise pandas statistics.
      2. Runs a FAISS similarity search for broader semantic context.
    Returns a merged, deduplicated list of Documents.
    """

    df: pd.DataFrame = Field(exclude=True)
    stats: dict = Field(exclude=True)
    vectorstore: FAISS = Field(exclude=True)
    k: int = Field(default=3)

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None) -> List[Document]:
        query_lower = query.lower()
        docs = []

        # --- Keyword-driven pandas lookups ---
        if any(w in query_lower for w in ["product", "widget", "item"]):
            docs.append(self._product_doc())

        if any(w in query_lower for w in ["region", "north", "south", "east", "west", "area"]):
            docs.append(self._region_doc())

        if any(w in query_lower for w in ["year", "annual", "2022", "2023", "2024"]):
            docs.append(self._yearly_doc())

        if any(w in query_lower for w in ["quarter", "q1", "q2", "q3", "q4"]):
            docs.append(self._quarterly_doc())

        if any(w in query_lower for w in ["month", "monthly", "january", "february",
                                           "march", "april", "may", "june", "july",
                                           "august", "september", "october", "november", "december"]):
            docs.append(self._monthly_doc())

        if any(w in query_lower for w in ["age", "gender", "male", "female",
                                           "demographic", "segment", "customer"]):
            docs.append(self._demographics_doc())

        if any(w in query_lower for w in ["satisfaction", "rating", "score", "happy"]):
            docs.append(self._satisfaction_doc())

        if any(w in query_lower for w in ["overall", "total", "summary", "average",
                                           "mean", "median", "std", "statistic"]):
            docs.append(self._overall_doc())

        # --- FAISS semantic search for remaining context ---
        vector_docs = self.vectorstore.similarity_search(query, k=self.k)
        docs.extend(vector_docs)

        # Deduplicate by page_content
        seen = set()
        unique_docs = []
        for doc in docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)

        return unique_docs

    # --- Individual stat builders ---

    def _overall_doc(self) -> Document:
        o = self.stats["overall"]
        return Document(
            page_content=(
                f"Overall Sales Statistics: total=${o['total_sales']:,}, "
                f"mean=${o['mean_sales']:,}, median=${o['median_sales']:,}, "
                f"std=${o['std_sales']:,}, min=${o['min_sales']:,}, max=${o['max_sales']:,}."
            ),
            metadata={"source": "pandas", "topic": "overall"}
        )

    def _product_doc(self) -> Document:
        by_product = self.stats["by_product"]
        lines = "\n".join(
            f"  {p}: total=${v['total']:,.0f}, mean=${v['mean']:,.2f}, "
            f"median=${v['median']:,.2f}, count={v['count']}"
            for p, v in by_product.items()
        )
        return Document(
            page_content=f"Sales by Product:\n{lines}",
            metadata={"source": "pandas", "topic": "product"}
        )

    def _region_doc(self) -> Document:
        by_region = self.stats["by_region"]
        lines = "\n".join(
            f"  {r}: total=${v['total']:,.0f}, mean=${v['mean']:,.2f}, "
            f"median=${v['median']:,.2f}, count={v['count']}"
            for r, v in by_region.items()
        )
        return Document(
            page_content=f"Sales by Region:\n{lines}",
            metadata={"source": "pandas", "topic": "region"}
        )

    def _yearly_doc(self) -> Document:
        by_year = self.stats["by_year"]
        lines = "\n".join(
            f"  {yr}: total=${v['total']:,.0f}, mean=${v['mean']:,.2f}"
            for yr, v in by_year.items()
        )
        return Document(
            page_content=f"Annual Sales:\n{lines}",
            metadata={"source": "pandas", "topic": "annual"}
        )

    def _quarterly_doc(self) -> Document:
        rows = self.stats["by_quarter"]
        lines = "\n".join(
            f"  Q{r['Quarter']} {r['Year']}: total=${r['total']:,.0f}"
            for r in rows
        )
        return Document(
            page_content=f"Quarterly Sales:\n{lines}",
            metadata={"source": "pandas", "topic": "quarterly"}
        )

    def _monthly_doc(self) -> Document:
        by_month = self.stats["by_month"]
        lines = "\n".join(
            f"  {m}: total=${v['total']:,.0f}, mean=${v['mean']:,.2f}"
            for m, v in by_month.items()
        )
        return Document(
            page_content=f"Monthly Sales (all years combined):\n{lines}",
            metadata={"source": "pandas", "topic": "monthly"}
        )

    def _demographics_doc(self) -> Document:
        d = self.stats["demographics"]
        age_lines = "\n".join(
            f"  {grp}: total=${v['total']:,.0f}, mean=${v['mean']:,.2f}"
            for grp, v in d["sales_by_age_group"].items()
        )
        gender_lines = "\n".join(
            f"  {g}: total=${v['total']:,.0f}, mean=${v['mean']:,.2f}"
            for g, v in d["sales_by_gender"].items()
        )
        return Document(
            page_content=(
                f"Customer Demographics:\n"
                f"Avg age={d['age_mean']} (std={d['age_std']}), "
                f"gender count={d['gender_distribution']}\n"
                f"Sales by age group:\n{age_lines}\n"
                f"Sales by gender:\n{gender_lines}"
            ),
            metadata={"source": "pandas", "topic": "demographics"}
        )

    def _satisfaction_doc(self) -> Document:
        s = self.stats["satisfaction"]
        p_lines = "\n".join(f"  {p}: {sc}/5.0" for p, sc in s["by_product"].items())
        r_lines = "\n".join(f"  {r}: {sc}/5.0" for r, sc in s["by_region"].items())
        return Document(
            page_content=(
                f"Customer Satisfaction:\n"
                f"Overall mean={s['overall_mean']}/5.0 (median={s['overall_median']}, std={s['overall_std']})\n"
                f"By product:\n{p_lines}\n"
                f"By region:\n{r_lines}"
            ),
            metadata={"source": "pandas", "topic": "satisfaction"}
        )


def build_retriever(k: int = 3) -> PandasStatsRetriever:
    """Load data, stats, and vector store; return a ready-to-use retriever."""
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

    df = load_data()
    stats = compute_summary_statistics(df)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True
    )
    return PandasStatsRetriever(df=df, stats=stats, vectorstore=vectorstore, k=k)
