import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import json
    import re
    from collections import Counter
    from datetime import datetime

    import polars as pl
    import plotly.graph_objects as go
    import plotly.express as px

    mo.md("# E-Commerce Data Analysis Dashboard")
    return Counter, datetime, go, json, mo, pl, px, re


@app.cell
def _(mo):
    intro = mo.md(
        """
    ## Loading Product and Review Data

    This notebook analyzes:
    - Top electronics products (up to 1000)
    - Top reviews for those products

    Use the tabs to navigate sections.
    """
    )
    return (intro,)


@app.cell
def _(json, pl):
    # Load product data
    products: list[dict] = []
    with open("data/top_1000_products.jsonl", "r") as fp:
        for line in fp:
            products.append(json.loads(line.strip()))

    products_df = pl.DataFrame(products)

    # Standardize a reviews-count column
    products_df = products_df.with_columns(
        pl.when(pl.col("review_count").is_not_null())
        .then(pl.col("review_count"))
        .otherwise(pl.col("rating_number"))
        .alias("num_reviews")
    )

    # Load review data
    reviews: list[dict] = []
    with open("data/100_top_reviews_of_the_top_1000_products.jsonl", "r") as fp:
        for line in fp:
            reviews.append(json.loads(line.strip()))

    reviews_df_raw = pl.DataFrame(reviews)

    # Add datetime column when timestamp exists (ms epoch)
    if "timestamp" in reviews_df_raw.columns:
        reviews_df_raw = reviews_df_raw.with_columns(
            pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("review_date")
        )

    return products_df, reviews_df_raw


@app.cell
def _(mo, products_df, reviews_df_raw):
    # Calculate date range for reviews
    if "review_date" in reviews_df_raw.columns:
        min_date = reviews_df_raw["review_date"].min()
        max_date = reviews_df_raw["review_date"].max()
        try:
            date_range = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
        except AttributeError:
            date_range = "N/A"
    else:
        date_range = "N/A"

    products_count = products_df.height
    reviews_count = reviews_df_raw.height
    avg_reviews_per_product = (
        reviews_count / products_count if products_count > 0 else float("nan")
    )

    overview_card = mo.callout(
        f"""
        - **Products loaded:** {products_count:,}
        - **Reviews loaded:** {reviews_count:,}
        - **Average reviews per product:** {avg_reviews_per_product:.1f}
        - **Date range of reviews:** {date_range}
        """,
        kind="info",
    )
    mo.vstack([mo.md("## Dataset Overview"), overview_card], align="stretch", gap=0.5)
    return avg_reviews_per_product, date_range, products_count, reviews_count


@app.cell
def _(mo):
    mo.md("""## Top Products Analysis""")
    return


@app.cell
def _(mo, pl, products_df):
    safe_title = pl.col("title").cast(pl.Utf8, strict=False).fill_null("")

    # Top by number of reviews
    top_by_reviews = (
        products_df
        .sort("num_reviews", descending=True)
        .head(10)
        .select(["title", "num_reviews", "average_rating"])
        .with_columns(safe_title.alias("title"))
        .with_columns(pl.col("title").str.slice(0, 80) + "...")
    )

    # Highest rated with minimum engagement
    top_by_rating = (
        products_df
        .filter(pl.col("num_reviews") >= 100)
        .sort("average_rating", descending=True)
        .head(10)
        .select(["title", "average_rating", "num_reviews"])
        .with_columns(safe_title.alias("title"))
        .with_columns(pl.col("title").str.slice(0, 80) + "...")
    )

    mo.tabs({
        "Most Reviewed": mo.ui.table(top_by_reviews.to_pandas(), pagination=False),
        "Highest Rated (>=100)": mo.ui.table(top_by_rating.to_pandas(), pagination=False),
    })
    return top_by_rating, top_by_reviews


@app.cell
def _(mo):
    mo.md("""## Rating Analysis""")
    return


@app.cell
def _(go, mo, pl, products_df, reviews_df_raw):
    # Rating distribution for products
    product_ratings = (
        products_df
        .group_by("average_rating")
        .agg(pl.len().alias("count"))
        .sort("average_rating")
    )

    # Rating distribution for individual reviews
    review_ratings = (
        reviews_df_raw
        .group_by("rating")
        .agg(pl.len().alias("count"))
        .sort("rating")
    )

    fig_ratings = go.Figure()

    fig_ratings.add_trace(
        go.Bar(
            x=product_ratings["average_rating"].to_list(),
            y=product_ratings["count"].to_list(),
            name="Product Average Ratings",
            marker_color="gold",
        )
    )

    fig_ratings.add_trace(
        go.Bar(
            x=review_ratings["rating"].to_list(),
            y=review_ratings["count"].to_list(),
            name="Individual Review Ratings",
            marker_color="orange",
            opacity=0.7,
        )
    )

    fig_ratings.update_layout(
        title="Rating Distributions",
        xaxis_title="Rating",
        yaxis_title="Count",
        barmode="group",
        height=420,
    )

    mo.ui.plotly(fig_ratings)
    return fig_ratings, product_ratings, review_ratings


@app.cell
def _(mo, products_df):
    # Correlation between engagement and ratings (via pandas for simplicity)
    df = products_df.select(["num_reviews", "average_rating"]).to_pandas()
    correlation = df["num_reviews"].corr(df["average_rating"]) if len(df) > 0 else float("nan")

    mo.md(
        f"""
    ### Correlation Analysis

    **Correlation between Review Count and Average Rating:** {correlation:.3f}

    This shows {'a positive' if correlation > 0 else 'a negative' if correlation < 0 else 'no'} correlation.
    """
    )
    return correlation


@app.cell
def _(mo):
    mo.md("""## Review Text Analysis""")
    return


@app.cell
def _(Counter, pl, re, reviews_df_raw):
    # Analyze review text lengths
    reviews_df = reviews_df_raw.with_columns(
        [
            pl.col("text").cast(pl.Utf8, strict=False).fill_null("").str.len_chars().alias("text_length"),
            pl.col("title").cast(pl.Utf8, strict=False).fill_null("").str.len_chars().alias("title_length"),
        ]
    )

    # Helpful votes analysis
    reviews_with_votes = (
        reviews_df.filter(pl.col("helpful_vote").cast(pl.Int64, strict=False).fill_null(0) > 0)
        if "helpful_vote" in reviews_df.columns
        else pl.DataFrame({"helpful_vote": []})
    )

    # Most common words in review titles (simple approach)
    all_titles = reviews_df["title"].fill_null("").to_list() if "title" in reviews_df.columns else []
    all_words = " ".join(all_titles).lower()
    words = re.findall(r"\b\w+\b", all_words)

    # Filter out common stop words
    stop_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "it",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "we",
        "they",
        "my",
        "your",
    }

    filtered_words = [w for w in words if w not in stop_words and len(w) > 2]
    word_freq = Counter(filtered_words).most_common(20)

    return reviews_df, reviews_with_votes, word_freq


@app.cell
def _(mo, pl, reviews_df, reviews_with_votes, word_freq):
    import pandas as pd_stats  # unique alias to avoid redefinition across cells

    avg_text_len = (
        reviews_df["text_length"].mean() if "text_length" in reviews_df.columns else float("nan")
    )
    avg_title_len = (
        reviews_df["title_length"].mean() if "title_length" in reviews_df.columns else float("nan")
    )

    mo.md(
        f"""
    ### Review Text Statistics

    - **Average review length:** {avg_text_len:.0f} characters
    - **Average title length:** {avg_title_len:.0f} characters
    - **Reviews with helpful votes:** {len(reviews_with_votes):,} ({len(reviews_with_votes)/len(reviews_df)*100:.1f}% if len(reviews_df) > 0 else 0.0)
    - **Average helpful votes (when > 0):** {(reviews_with_votes['helpful_vote'].mean() if len(reviews_with_votes) > 0 else 0.0):.1f}

    ### Most Common Words in Review Titles
    {mo.ui.table(pd_stats.DataFrame(word_freq[:10], columns=["Word", "Frequency"]), pagination=False)}
    """
    )
    return pd_stats


@app.cell
def _(mo):
    mo.md("""## Temporal Analysis""")
    return


@app.cell
def _(mo, pl, px, reviews_df):
    if "review_date" in reviews_df.columns:
        reviews_df_with_month = reviews_df.with_columns(
            pl.col("review_date").dt.strftime("%Y-%m").alias("year_month")
        )
        reviews_by_month = reviews_df_with_month.group_by("year_month").agg(pl.len().alias("count")).sort("year_month")

        fig_timeline = px.line(
            reviews_by_month.tail(24).to_pandas(),
            x="year_month",
            y="count",
            title="Reviews Over Time (Last 24 Months)",
            labels={"year_month": "Month", "count": "Number of Reviews"},
        )
        fig_timeline.update_layout(height=400)
        display_timeline = fig_timeline
    else:
        display_timeline = mo.md("*Timeline analysis not available - no timestamp data*")

    display_timeline
    return display_timeline


@app.cell
def _(mo):
    mo.md("""## Category Analysis""")
    return


@app.cell
def _(Counter, mo, pl, products_df):
    import pandas as pd_cat  # unique alias to avoid redefinition

    # Analyze main categories
    all_categories: list[str] = []
    if "main_category" in products_df.columns:
        all_categories.extend(
            [c for c in products_df["main_category"].to_list() if isinstance(c, str) and c]
        )

    category_counts = Counter(all_categories).most_common(15)

    if category_counts:
        category_df = pd_cat.DataFrame(category_counts, columns=["Category", "Count"])
        mo.md(
            f"""
        ### Top Product Categories
        {mo.ui.table(category_df, pagination=False)}
        """
        )
    else:
        mo.md("*Category data not available*")

    return category_counts


@app.cell
def _(go, pl, products_df):
    # Review Count vs Rating scatter plot
    fig_scatter = go.Figure()
    scatter_data = products_df.select(["title", "num_reviews", "average_rating"]) if products_df.height else products_df

    if scatter_data.height:
        fig_scatter.add_trace(
            go.Scatter(
                x=scatter_data["num_reviews"].to_list(),
                y=scatter_data["average_rating"].to_list(),
                mode="markers",
                marker=dict(
                    size=8,
                    color=scatter_data["num_reviews"].to_list(),
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Review Count"),
                    line=dict(width=1, color="white"),
                ),
                text=[(t or "")[:40] + "..." for t in scatter_data["title"].to_list()],
                hovertemplate="<b>%{text}</b><br>Reviews: %{x}<br>Rating: %{y:.2f}<extra></extra>",
            )
        )

    fig_scatter.update_layout(
        title="Review Count vs Rating",
        xaxis_title="Number of Reviews",
        yaxis_title="Average Rating",
        height=480,
    )

    fig_scatter
    return fig_scatter


@app.cell
def _(mo):
    mo.md("""## Business Insights""")
    return


@app.cell
def _(mo, pl, products_df, reviews_df_raw):
    total_reviews = reviews_df_raw.height
    avg_rating = products_df["average_rating"].mean() if products_df.height else float("nan")
    total_products = products_df.height

    # High engagement: top quartile of reviews and rating >= 4.0
    review_75th = products_df["num_reviews"].quantile(0.75) if products_df.height else 0
    high_engagement = products_df.filter(
        (pl.col("num_reviews") > review_75th) & (pl.col("average_rating") >= 4.0)
    )

    review_segments = {
        "Low engagement (< 100 reviews)": products_df.filter(pl.col("num_reviews") < 100).height,
        "Medium engagement (100-500)": products_df.filter(
            (pl.col("num_reviews") >= 100) & (pl.col("num_reviews") < 500)
        ).height,
        "High engagement (500-1000)": products_df.filter(
            (pl.col("num_reviews") >= 500) & (pl.col("num_reviews") < 1000)
        ).height,
        "Very high engagement (1000+)": products_df.filter(pl.col("num_reviews") >= 1000).height,
    }

    mo.md(
        f"""
    ### Key Business Metrics

    - Total Products Analyzed: **{total_products:,}**
    - Total Reviews Analyzed: **{total_reviews:,}**
    - Average Product Rating: **{avg_rating:.2f}**
    - High Engagement Products: **{high_engagement.height}** (high reviews + rating ≥ 4.0)

    **Review Engagement Segmentation:**
    - Low engagement (< 100 reviews): **{review_segments['Low engagement (< 100 reviews)']:,}**
    - Medium engagement (100-500): **{review_segments['Medium engagement (100-500)']:,}**
    - High engagement (500-1000): **{review_segments['High engagement (500-1000)']:,}**
    - Very high engagement (1000+): **{review_segments['Very high engagement (1000+)']:,}**
    """
    )
    return avg_rating, high_engagement, review_segments, total_products, total_reviews


@app.cell
def _(mo):
    mo.md("""## 🔍 Product Search & Details""")
    return


@app.cell
def _(mo):
    search_term = mo.ui.text(placeholder="Search for a product...", label="Product Search")

    mo.md(
        f"""
    ### Search Products
    {search_term}
    """
    )
    return (search_term,)


@app.cell
def _(mo, pl, products_df, reviews_df_raw, search_term):
    first_product = None
    product_reviews = None
    search_results = None

    view = None
    if search_term.value:
        query = search_term.value.strip().lower()
        search_results = (
            products_df
            .filter(
                pl.col("title")
                .cast(pl.Utf8, strict=False)
                .fill_null("")
                .str.to_lowercase()
                .str.contains(query, literal=True)
            )
            .head(10)
        )

        if search_results.height > 0:
            first_product = search_results.row(0, named=True)
            product_reviews = (
                reviews_df_raw.filter(pl.col("parent_asin") == first_product.get("parent_asin")).select(["rating", "title", "helpful_vote"]).head(5)
            )
            view = mo.md(
                f"""
            ### Search Results for "{search_term.value}"

            Found **{search_results.height}** matching products

            **Top Match:** {(first_product.get('title') or '')[:100]}...
            - Rating: **{first_product.get('average_rating', float('nan')):.1f}** ⭐
            - Reviews: **{first_product.get('num_reviews', 0):,}**

            **Sample Reviews:**
            {mo.ui.table(product_reviews.to_pandas(), pagination=False)}
            
            **Matches (up to 10):**
            {mo.ui.table(search_results.select(["title", "average_rating", "num_reviews"]).to_pandas(), pagination=False)}
            """
            )
        else:
            view = mo.md(f"No products found matching '{search_term.value}'")
    else:
        view = mo.md("*Enter a search term above to find products*")
    mo.tabs({
        "Search": view,
        "Stats": mo.md(
            f"""
            **Products:** {products_df.height:,} • **Reviews:** {reviews_df_raw.height:,}
            """
        ),
    })

    return first_product, product_reviews, search_results


@app.cell
def _(mo):
    mo.md(
        """
    ## Summary & Recommendations

    - Product quality is broadly high (ratings clustered above 4.0)
    - Products with many reviews tend to maintain solid ratings
    - Consider promoting high-rated products with mid-tier review counts
    - Mine review text for recurring themes to inform improvements
    """
    )
    return


if __name__ == "__main__":
    app.run()


