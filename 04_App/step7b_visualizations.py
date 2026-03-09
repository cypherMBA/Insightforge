"""
Step 7b: Data Visualizations
Produces all required charts as Plotly figures (compatible with Streamlit).
Charts:
  1. Sales trend over time (monthly)
  2. Product performance comparison
  3. Regional sales analysis
  4. Customer demographics & segmentation
  5. Customer satisfaction heatmap (product x region)
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from step1_data_preparation import load_data

PALETTE = px.colors.qualitative.Set2


# ---------------------------------------------------------------------------
# 1. Sales Trend Over Time
# ---------------------------------------------------------------------------

def plot_sales_trend(df: pd.DataFrame) -> go.Figure:
    """Monthly total sales line chart with year-over-year comparison."""
    monthly = (
        df.groupby(["Year", "Month", "Month_Name"])["Sales"]
        .sum()
        .reset_index()
        .sort_values(["Year", "Month"])
    )
    monthly["Period"] = monthly["Month_Name"] + " " + monthly["Year"].astype(str)

    fig = px.line(
        monthly,
        x="Month",
        y="Sales",
        color="Year",
        markers=True,
        title="Monthly Sales Trend by Year",
        labels={"Sales": "Total Sales ($)", "Month": "Month"},
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(
        xaxis=dict(tickmode="array", tickvals=list(range(1, 13)),
                   ticktext=["Jan","Feb","Mar","Apr","May","Jun",
                              "Jul","Aug","Sep","Oct","Nov","Dec"]),
        hovermode="x unified",
        legend_title="Year",
    )
    return fig


def plot_quarterly_sales(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of quarterly sales per year."""
    quarterly = (
        df.groupby(["Year", "Quarter"])["Sales"]
        .sum()
        .reset_index()
    )
    quarterly["Quarter_Label"] = "Q" + quarterly["Quarter"].astype(str)

    fig = px.bar(
        quarterly,
        x="Quarter_Label",
        y="Sales",
        color="Year",
        barmode="group",
        title="Quarterly Sales by Year",
        labels={"Sales": "Total Sales ($)", "Quarter_Label": "Quarter"},
        color_discrete_sequence=PALETTE,
        text_auto=".2s",
    )
    fig.update_traces(textposition="outside")
    return fig


# ---------------------------------------------------------------------------
# 2. Product Performance
# ---------------------------------------------------------------------------

def plot_product_performance(df: pd.DataFrame) -> go.Figure:
    """Side-by-side subplots: total sales and average satisfaction per product."""
    product_sales = df.groupby("Product")["Sales"].sum().reset_index()
    product_sat = df.groupby("Product")["Customer_Satisfaction"].mean().reset_index()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Total Sales by Product", "Avg Satisfaction by Product"),
    )

    fig.add_trace(
        go.Bar(
            x=product_sales["Product"],
            y=product_sales["Sales"],
            marker_color=PALETTE[:4],
            name="Total Sales",
            text=product_sales["Sales"].apply(lambda v: f"${v:,.0f}"),
            textposition="outside",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Bar(
            x=product_sat["Product"],
            y=product_sat["Customer_Satisfaction"].round(2),
            marker_color=PALETTE[4:8],
            name="Avg Satisfaction",
            text=product_sat["Customer_Satisfaction"].round(2),
            textposition="outside",
        ),
        row=1, col=2,
    )

    fig.update_layout(title_text="Product Performance Overview", showlegend=False)
    fig.update_yaxes(title_text="Total Sales ($)", row=1, col=1)
    fig.update_yaxes(title_text="Satisfaction (1–5)", row=1, col=2)
    return fig


def plot_product_region_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap of total sales for each product–region combination."""
    pivot = df.pivot_table(values="Sales", index="Product", columns="Region", aggfunc="sum")

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="Blues",
            text=[[f"${v:,.0f}" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            hovertemplate="Product: %{y}<br>Region: %{x}<br>Sales: %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Sales Heatmap: Product × Region",
        xaxis_title="Region",
        yaxis_title="Product",
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Regional Analysis
# ---------------------------------------------------------------------------

def plot_regional_sales(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of total sales by region."""
    regional = df.groupby("Region")["Sales"].sum().reset_index().sort_values("Sales")

    fig = px.bar(
        regional,
        x="Sales",
        y="Region",
        orientation="h",
        title="Total Sales by Region",
        labels={"Sales": "Total Sales ($)"},
        color="Region",
        color_discrete_sequence=PALETTE,
        text_auto=".2s",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False)
    return fig


def plot_regional_trend(df: pd.DataFrame) -> go.Figure:
    """Line chart: monthly sales per region."""
    monthly_region = (
        df.groupby(["Year", "Month", "Region"])["Sales"]
        .sum()
        .reset_index()
        .sort_values(["Year", "Month"])
    )
    monthly_region["Period"] = (
        monthly_region["Year"].astype(str) + "-"
        + monthly_region["Month"].astype(str).str.zfill(2)
    )

    fig = px.line(
        monthly_region,
        x="Period",
        y="Sales",
        color="Region",
        title="Monthly Sales Trend by Region",
        labels={"Sales": "Total Sales ($)", "Period": "Month"},
        color_discrete_sequence=PALETTE,
    )
    fig.update_xaxes(tickangle=45, nticks=20)
    return fig


# ---------------------------------------------------------------------------
# 4. Customer Demographics & Segmentation
# ---------------------------------------------------------------------------

def plot_gender_distribution(df: pd.DataFrame) -> go.Figure:
    """Pie chart of customer gender distribution."""
    gender_counts = df["Customer_Gender"].value_counts().reset_index()
    gender_counts.columns = ["Gender", "Count"]

    fig = px.pie(
        gender_counts,
        names="Gender",
        values="Count",
        title="Customer Gender Distribution",
        color_discrete_sequence=PALETTE,
        hole=0.4,
    )
    fig.update_traces(textinfo="percent+label")
    return fig


def plot_age_group_sales(df: pd.DataFrame) -> go.Figure:
    """Bar chart of total sales by customer age group."""
    df2 = df.copy()
    df2["Age_Group"] = pd.cut(
        df2["Customer_Age"],
        bins=[18, 30, 45, 60, 70],
        labels=["18–30", "31–45", "46–60", "61–70"],
    )
    age_sales = df2.groupby("Age_Group", observed=True)["Sales"].sum().reset_index()

    fig = px.bar(
        age_sales,
        x="Age_Group",
        y="Sales",
        title="Total Sales by Customer Age Group",
        labels={"Sales": "Total Sales ($)", "Age_Group": "Age Group"},
        color="Age_Group",
        color_discrete_sequence=PALETTE,
        text_auto=".2s",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False)
    return fig


def plot_gender_sales_by_product(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart: sales by gender for each product."""
    gender_product = df.groupby(["Product", "Customer_Gender"])["Sales"].sum().reset_index()

    fig = px.bar(
        gender_product,
        x="Product",
        y="Sales",
        color="Customer_Gender",
        barmode="group",
        title="Sales by Product and Gender",
        labels={"Sales": "Total Sales ($)", "Customer_Gender": "Gender"},
        color_discrete_sequence=PALETTE,
        text_auto=".2s",
    )
    fig.update_traces(textposition="outside")
    return fig


# ---------------------------------------------------------------------------
# 5. Customer Satisfaction
# ---------------------------------------------------------------------------

def plot_satisfaction_by_product_region(df: pd.DataFrame) -> go.Figure:
    """Heatmap of average customer satisfaction by product and region."""
    pivot = df.pivot_table(
        values="Customer_Satisfaction",
        index="Product",
        columns="Region",
        aggfunc="mean",
    ).round(2)

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="RdYlGn",
            zmin=1,
            zmax=5,
            text=pivot.values,
            texttemplate="%{text:.2f}",
            hovertemplate="Product: %{y}<br>Region: %{x}<br>Satisfaction: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Avg Customer Satisfaction: Product × Region",
        xaxis_title="Region",
        yaxis_title="Product",
    )
    return fig


# ---------------------------------------------------------------------------
# Convenience: return all figures as a dict
# ---------------------------------------------------------------------------

def build_all_figures(df: pd.DataFrame) -> dict:
    return {
        "sales_trend": plot_sales_trend(df),
        "quarterly_sales": plot_quarterly_sales(df),
        "product_performance": plot_product_performance(df),
        "product_region_heatmap": plot_product_region_heatmap(df),
        "regional_sales": plot_regional_sales(df),
        "regional_trend": plot_regional_trend(df),
        "gender_distribution": plot_gender_distribution(df),
        "age_group_sales": plot_age_group_sales(df),
        "gender_sales_by_product": plot_gender_sales_by_product(df),
        "satisfaction_heatmap": plot_satisfaction_by_product_region(df),
    }


if __name__ == "__main__":
    import plotly.io as pio
    df = load_data()
    figs = build_all_figures(df)
    for name, fig in figs.items():
        print(f"Rendering: {name}")
        pio.show(fig)
