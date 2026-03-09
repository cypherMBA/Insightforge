"""
Step 1: Data Preparation
Loads and explores the sales dataset, computes key summary statistics
that will feed into the RAG knowledge base.
"""

import pandas as pd
import numpy as np
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "03_Sales_Data", "sales_data.csv")


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Month_Name"] = df["Date"].dt.strftime("%B")
    df["Quarter"] = df["Date"].dt.quarter
    return df


def explore_data(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape          : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Date range     : {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Products       : {sorted(df['Product'].unique().tolist())}")
    print(f"Regions        : {sorted(df['Region'].unique().tolist())}")
    print(f"Genders        : {sorted(df['Customer_Gender'].unique().tolist())}")
    print(f"Age range      : {df['Customer_Age'].min()} – {df['Customer_Age'].max()}")
    print(f"Sales range    : {df['Sales'].min()} – {df['Sales'].max()}")
    print(f"Missing values :\n{df.isnull().sum()}")
    print()


def compute_summary_statistics(df: pd.DataFrame) -> dict:
    """Compute all statistics needed for the knowledge base."""

    stats = {}

    # --- Overall sales ---
    stats["overall"] = {
        "total_sales": int(df["Sales"].sum()),
        "mean_sales": round(df["Sales"].mean(), 2),
        "median_sales": round(df["Sales"].median(), 2),
        "std_sales": round(df["Sales"].std(), 2),
        "min_sales": int(df["Sales"].min()),
        "max_sales": int(df["Sales"].max()),
    }

    # --- Sales by year ---
    stats["by_year"] = (
        df.groupby("Year")["Sales"]
        .agg(total="sum", mean="mean", median="median")
        .round(2)
        .to_dict(orient="index")
    )

    # --- Sales by month (aggregated across all years) ---
    stats["by_month"] = (
        df.groupby("Month_Name")["Sales"]
        .agg(total="sum", mean="mean")
        .round(2)
        .to_dict(orient="index")
    )

    # --- Sales by quarter ---
    stats["by_quarter"] = (
        df.groupby(["Year", "Quarter"])["Sales"]
        .sum()
        .reset_index()
        .rename(columns={"Sales": "total"})
        .to_dict(orient="records")
    )

    # --- Sales by product ---
    stats["by_product"] = (
        df.groupby("Product")["Sales"]
        .agg(total="sum", mean="mean", median="median", count="count")
        .round(2)
        .to_dict(orient="index")
    )

    # --- Sales by region ---
    stats["by_region"] = (
        df.groupby("Region")["Sales"]
        .agg(total="sum", mean="mean", median="median", count="count")
        .round(2)
        .to_dict(orient="index")
    )

    # --- Product x Region ---
    stats["product_region"] = (
        df.groupby(["Product", "Region"])["Sales"]
        .sum()
        .reset_index()
        .rename(columns={"Sales": "total"})
        .to_dict(orient="records")
    )

    # --- Customer demographics ---
    stats["demographics"] = {
        "age_mean": round(df["Customer_Age"].mean(), 2),
        "age_median": float(df["Customer_Age"].median()),
        "age_std": round(df["Customer_Age"].std(), 2),
        "gender_distribution": df["Customer_Gender"].value_counts().to_dict(),
        "sales_by_gender": (
            df.groupby("Customer_Gender")["Sales"]
            .agg(total="sum", mean="mean")
            .round(2)
            .to_dict(orient="index")
        ),
        "sales_by_age_group": (
            df.assign(
                Age_Group=pd.cut(
                    df["Customer_Age"],
                    bins=[18, 30, 45, 60, 70],
                    labels=["18-30", "31-45", "46-60", "61-70"],
                )
            )
            .groupby("Age_Group", observed=True)["Sales"]
            .agg(total="sum", mean="mean")
            .round(2)
            .to_dict(orient="index")
        ),
    }

    # --- Customer satisfaction ---
    stats["satisfaction"] = {
        "overall_mean": round(df["Customer_Satisfaction"].mean(), 2),
        "overall_median": round(df["Customer_Satisfaction"].median(), 2),
        "overall_std": round(df["Customer_Satisfaction"].std(), 2),
        "by_product": (
            df.groupby("Product")["Customer_Satisfaction"]
            .mean()
            .round(2)
            .to_dict()
        ),
        "by_region": (
            df.groupby("Region")["Customer_Satisfaction"]
            .mean()
            .round(2)
            .to_dict()
        ),
    }

    return stats


def print_summary(stats: dict) -> None:
    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    o = stats["overall"]
    print(f"\n[Overall Sales]")
    print(f"  Total   : ${o['total_sales']:,}")
    print(f"  Mean    : ${o['mean_sales']:,}")
    print(f"  Median  : ${o['median_sales']:,}")
    print(f"  Std Dev : ${o['std_sales']:,}")

    print(f"\n[Sales by Year]")
    for year, v in stats["by_year"].items():
        print(f"  {year}: Total=${v['total']:,.0f}, Mean=${v['mean']:,.2f}")

    print(f"\n[Sales by Product]")
    for product, v in stats["by_product"].items():
        print(f"  {product}: Total=${v['total']:,.0f}, Mean=${v['mean']:,.2f}")

    print(f"\n[Sales by Region]")
    for region, v in stats["by_region"].items():
        print(f"  {region}: Total=${v['total']:,.0f}, Mean=${v['mean']:,.2f}")

    print(f"\n[Customer Demographics]")
    d = stats["demographics"]
    print(f"  Avg Age : {d['age_mean']} (std={d['age_std']})")
    print(f"  Gender  : {d['gender_distribution']}")

    print(f"\n[Customer Satisfaction]")
    s = stats["satisfaction"]
    print(f"  Overall Mean : {s['overall_mean']} / 5.0")
    print(f"  By Product   : {s['by_product']}")
    print(f"  By Region    : {s['by_region']}")
    print()


if __name__ == "__main__":
    df = load_data()
    explore_data(df)
    stats = compute_summary_statistics(df)
    print_summary(stats)
    print("Data preparation complete.")
