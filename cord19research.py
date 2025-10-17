"""PlP program script for extracting data from a file in folder called metadata.csv,
preparing it,exploring, analyzing then visualization even using streamlit.
"""

# step 1: Imports & Display Settings ----------
# Install libraries if not installed (run separately in terminal):
# pip install pandas matplotlib seaborn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import string
import os

# Plot style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

# Make outputs easier to read in notebooks
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

# step 2: Function to load CSV from local path or URL ----------
def load_metadata(path_or_url, nrows=None):
    """
    Load metadata CSV into a pandas DataFrame.
    - path_or_url: local file path or a URL to a CSV
    - nrows: optional, load only first n rows (useful for GitHub/sample)
    Returns DataFrame or raises informative exceptions.
    """
    try:
        print(f"Loading data from: {path_or_url}")
        df = pd.read_csv(path_or_url, low_memory=False, nrows=nrows)
        print("✅ Loaded successfully.")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at: {path_or_url}. Put metadata.csv in the same folder or give full path.")
    except pd.errors.EmptyDataError:
        raise ValueError("The CSV is empty or unreadable.")
    except Exception as e:
        raise RuntimeError(f"Error loading CSV: {e}")
      
# step 3: Quick exploration 
def explore_df(df):
    """Show head, shape, dtypes and missing-value summary."""
    print("\n--- First 5 rows ---")
    display(df.head())
    print("\n--- Dimensions ---")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("\n--- Data types ---")
    print(df.dtypes)
    print("\n--- Missing values (top columns) ---")
    print(df.isnull().sum().sort_values(ascending=False).head(20))
  
# step 4: Cleaning & Preparation ----------
def clean_and_prepare(df):
    """
    - Finds a probable date column and converts it to datetime
    - Extracts 'year' column for time analysis
    - Creates 'abstract_word_count' if abstract exists
    - Tries to find a journal/source column among common names
    Returns cleaned DataFrame and column names used.
    """
    df2 = df.copy()

    # 1) Date column: try common names
    date_cols = ["publish_time", "publish_date", "date", "pub_date", "publication_date"]
    found_date = None
    for c in date_cols:
        if c in df2.columns:
            found_date = c
            break

    if found_date:
        print(f"Using date column: {found_date} -> converting to datetime.")
        df2[found_date] = pd.to_datetime(df2[found_date], errors="coerce")
        # create year column (useful for grouping)
        df2["year"] = df2[found_date].dt.year
    else:
        print("No obvious date column found. 'year' column will not be created.")

    # 2) Journal/source column: try common names
    journal_cols = ["journal", "journal_name", "journal-title", "publication", "source_x", "source"]
    found_journal = None
    for c in journal_cols:
        if c in df2.columns:
            found_journal = c
            break
    if found_journal:
        print(f"Using journal/source column: {found_journal}")
    else:
        print("No journal/source column found among common names.")

    # 3) Abstract word count
    if "abstract" in df2.columns:
        df2["abstract_word_count"] = df2["abstract"].fillna("").astype(str).map(lambda s: len(s.split()))
        print("Created 'abstract_word_count' from 'abstract' column.")
    else:
        print("No 'abstract' column found; skipping abstract_word_count.")

    # 4) Basic missing-value handling: drop rows with no title AND no abstract (not useful)
    if "title" in df2.columns or "abstract" in df2.columns:
        before = df2.shape[0]
        df2 = df2[~(df2.get("title", "").isnull() & df2.get("abstract", "").isnull())]
        after = df2.shape[0]
        print(f"Dropped {before - after} rows with no title and no abstract.")
    else:
        print("No title/abstract columns to check for blank rows.")

    return df2, found_date, found_journal


# step 5: Basic Analysis ----------
def basic_analysis(df, date_col=None, journal_col=None, top_n=10):
    """
    Performs:
    - Counts by year (if 'year' exists)
    - Top journals (if journal_col provided)
    - Word frequency in titles
    Returns summary dict for downstream plotting.
    """
    summary = {}

    # A) Papers by year
    if "year" in df.columns:
        counts_by_year = df["year"].value_counts(dropna=True).sort_index()
        print("\nPapers by year:")
        print(counts_by_year)
        summary["counts_by_year"] = counts_by_year
    else:
        print("\nNo 'year' column available for time-series analysis.")

    # B) Top journals
    if journal_col:
        top_journals = df[journal_col].fillna("Unknown").value_counts().head(top_n)
        print(f"\nTop {top_n} journals/sources:")
        print(top_journals)
        summary["top_journals"] = top_journals
    else:
        print("\nNo journal/source column provided for top journals analysis.")

    # C) Word frequency in titles (simple)
    if "title" in df.columns:
        titles = df["title"].dropna().astype(str).str.lower()
        # simple tokenization: split on whitespace and strip punctuation
        translator = str.maketrans("", "", string.punctuation)
        all_words = titles.map(lambda t: t.translate(translator).split()).sum()
        counter = Counter(all_words)
        # remove some common English stopwords (small list)
        stopwords = set(["the","and","of","in","to","a","for","on","with","by","from","is","that","are","as","an"])
        for sw in stopwords:
            if sw in counter:
                del counter[sw]
        top_words = counter.most_common(30)
        print("\nTop words in titles (sample):")
        print(top_words[:20])
        summary["top_title_words"] = top_words
    else:
        print("\nNo 'title' column to analyze words.")

    return summary
# step 6: Visualizations ----------
def create_visualizations(df, summary, date_col=None, journal_col=None, output_dir="figs"):
    """
    Creates:
    - Line chart: publications over time (year)
    - Bar chart: top journals
    - Histogram: abstract_word_count (or other numeric col)
    - Scatter: two numeric columns (sepal/petal equivalent -> use abstract_word_count vs title length)
    Saves figures to output_dir and shows them.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) Line chart: publications over time
    if "counts_by_year" in summary:
        plt.figure()
        summary["counts_by_year"].plot(marker='o')
        plt.title("Publications by Year")
        plt.xlabel("Year")
        plt.ylabel("Number of papers")
        plt.tight_layout()
        fname = os.path.join(output_dir, "publications_by_year.png")
        plt.savefig(fname)
        plt.show()
        print(f"Saved line chart to {fname}")
    else:
        print("Skipping line chart (no year data).")

    # 2) Bar chart: top journals
    if "top_journals" in summary:
        plt.figure()
        summary["top_journals"].plot(kind="bar")
        plt.title("Top Publishing Journals / Sources")
        plt.xlabel("Journal / Source")
        plt.ylabel("Number of papers")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fname = os.path.join(output_dir, "top_journals.png")
        plt.savefig(fname)
        plt.show()
        print(f"Saved bar chart to {fname}")
    else:
        print("Skipping bar chart (no journal data).")

    # 3) Histogram: abstract_word_count
    if "abstract_word_count" in df.columns:
        plt.figure()
        df["abstract_word_count"].dropna().plot(kind="hist", bins=30)
        plt.title("Distribution of Abstract Word Count")
        plt.xlabel("Abstract word count")
        plt.tight_layout()
        fname = os.path.join(output_dir, "abstract_wordcount_hist.png")
        plt.savefig(fname)
        plt.show()
        print(f"Saved histogram to {fname}")
    else:
        print("Skipping histogram (abstract_word_count not found).")

    # 4) Scatter plot: abstract_word_count vs title length (if title exists)
    if "abstract_word_count" in df.columns and "title" in df.columns:
        plt.figure()
        title_len = df["title"].fillna("").astype(str).map(len)
        plt.scatter(title_len, df["abstract_word_count"], alpha=0.5)
        plt.title("Title Length vs Abstract Word Count")
        plt.xlabel("Title length (chars)")
        plt.ylabel("Abstract word count")
        plt.tight_layout()
        fname = os.path.join(output_dir, "titlelen_vs_abstractwords_scatter.png")
        plt.savefig(fname)
        plt.show()
        print(f"Saved scatter plot to {fname}")
    else:
        print("Skipping scatter plot (needs title and abstract_word_count).")

# step 7: Run the workflow ----------
def run_workflow(path_or_url, sample_rows=None):
    # Load
    df = load_metadata(path_or_url, nrows=sample_rows)
    # Explore
    explore_df(df)
    # Clean/prepare
    df_clean, date_col, journal_col = clean_and_prepare(df)
    # Analysis
    summary = basic_analysis(df_clean, date_col, journal_col, top_n=10)
    # Visualize (saves images to ./figs)
    create_visualizations(df_clean, summary, date_col, journal_col)
    # Return cleaned df and summary for further use
    return df_clean, summary

# Example usage:
# If metadata.csv is in same folder:
# cleaned_df, summary = run_workflow("metadata.csv", sample_rows=1000)
# OR, for a full file use:
# cleaned_df, summary = run_workflow("path/to/metadata.csv")

# app.py
"""
Simple Streamlit app to explore CORD-19 metadata.csv (sample .
pip install pandas matplotlib seaborn streamlit
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import string
import os

# UI / style
st.set_page_config(page_title="CORD-19 Metadata Explorer", layout="wide")
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 4)

# -------------------------
# Helper functions
# -------------------------
@st.cache_data
def load_csv(path_or_file, nrows=None):
    """
    Accept either a path (string) or an uploaded file-like object from streamlit.
    """
    try:
        if hasattr(path_or_file, "read"):  # file-like (uploaded)
            df = pd.read_csv(path_or_file, low_memory=False, nrows=nrows)
        else:
            df = pd.read_csv(path_or_file, low_memory=False, nrows=nrows)
        return df
    except Exception as e:
        raise RuntimeError(f"Error loading CSV: {e}")

def basic_clean_and_prepare(df):
    df = df.copy()

    # Find common date column and convert
    date_cols = ["publish_time", "publish_date", "date", "pub_date", "publication_date"]
    found_date = None
    for c in date_cols:
        if c in df.columns:
            found_date = c
            break
    if found_date:
        df[found_date] = pd.to_datetime(df[found_date], errors="coerce")
        df["year"] = df[found_date].dt.year

    # Find journal/source column
    journal_cols = ["journal", "journal_name", "journal-title", "publication", "source_x", "source"]
    found_journal = None
    for c in journal_cols:
        if c in df.columns:
            found_journal = c
            break

    # Abstract word count
    if "abstract" in df.columns:
        df["abstract_word_count"] = df["abstract"].fillna("").astype(str).map(lambda s: len(s.split()))

    # Drop rows with neither title nor abstract (not useful)
    if "title" in df.columns or "abstract" in df.columns:
        df = df[~(df.get("title", "").isnull() & df.get("abstract", "").isnull())]

    return df, found_date, found_journal

def compute_summaries(df, journal_col=None):
    summary = {}
    if "year" in df.columns:
        summary["counts_by_year"] = df["year"].value_counts(dropna=True).sort_index()
    if journal_col:
        summary["top_journals"] = df[journal_col].fillna("Unknown").value_counts().head(15)
    # Top words in titles (simple)
    if "title" in df.columns:
        titles = df["title"].dropna().astype(str).str.lower()
        translator = str.maketrans("", "", string.punctuation)
        all_words = titles.map(lambda t: t.translate(translator).split()).sum()
        counter = Counter(all_words)
        stopwords = set(["the","and","of","in","to","a","for","on","with","by","from","is","that","are","as","an"])
        for sw in stopwords:
            if sw in counter:
                del counter[sw]
        summary["top_title_words"] = counter.most_common(30)
    return summary

def plot_publications_by_year(counts_by_year):
    fig, ax = plt.subplots()
    counts_by_year.plot(marker="o", ax=ax)
    ax.set_title("Publications by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of papers")
    plt.tight_layout()
    return fig

def plot_top_journals(top_journals):
    fig, ax = plt.subplots()
    top_journals.plot(kind="bar", ax=ax)
    ax.set_title("Top Publishing Journals / Sources")
    ax.set_xlabel("Journal / Source")
    ax.set_ylabel("Number of papers")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig

def plot_abstract_wordcount_hist(df):
    fig, ax = plt.subplots()
    df["abstract_word_count"].dropna().plot(kind="hist", bins=30, ax=ax)
    ax.set_title("Distribution of Abstract Word Count")
    ax.set_xlabel("Abstract word count")
    plt.tight_layout()
    return fig

def plot_titlelen_vs_abstractwords(df):
    fig, ax = plt.subplots()
    title_len = df["title"].fillna("").astype(str).map(len)
    ax.scatter(title_len, df["abstract_word_count"], alpha=0.5)
    ax.set_title("Title length (chars) vs Abstract word count")
    ax.set_xlabel("Title length (chars)")
    ax.set_ylabel("Abstract word count")
    plt.tight_layout()
    return fig

# -------------------------
# Streamlit layout
# -------------------------
st.title("CORD-19 Metadata Explorer")
st.write("A simple interactive explorer for the `metadata.csv` file from the CORD-19 dataset.")
st.markdown("**Instructions:** Upload `metadata.csv` or provide a local path. Use a sample (n rows) for quick demo.")

col1, col2 = st.columns([2, 1])

with col2:
    st.header("Load options")
    uploaded_file = st.file_uploader("Upload metadata.csv (optional)", type=["csv"])
    local_path = st.text_input("Or enter local path to metadata.csv", value="")
    nrows = st.number_input("Load only first N rows (0 = all)", min_value=0, value=0, step=100)
    load_button = st.button("Load data")

# Only load when user clicks
if load_button:
    path_or_file = uploaded_file if uploaded_file is not None else (local_path if local_path.strip() else None)
    if path_or_file is None:
        st.error("Please upload a CSV file or specify a local path to metadata.csv.")
    else:
        try:
            df = load_csv(path_or_file, nrows=(None if nrows == 0 else int(nrows)))
            st.success(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
            st.subheader("Preview of data (first 5 rows)")
            st.dataframe(df.head())

            # Clean & prepare
            df_clean, date_col, journal_col = basic_clean_and_prepare(df)
            st.write(f"Detected date column: `{date_col}`" if date_col else "No date column detected.")
            st.write(f"Detected journal/source column: `{journal_col}`" if journal_col else "No journal/source column detected.")
            st.write(f"Rows after basic cleaning: {df_clean.shape[0]}")

            # Summaries
            summary = compute_summaries(df_clean, journal_col)

            # Show numeric summary
            st.subheader("Basic statistics (numeric columns)")
            st.write(df_clean.describe())

            # Publications by year
            if "counts_by_year" in summary:
                st.subheader("Publications by Year")
                fig_years = plot_publications_by_year(summary["counts_by_year"])
                st.pyplot(fig_years)

            # Top journals
            if "top_journals" in summary:
                st.subheader("Top Publishing Journals / Sources")
                fig_journals = plot_top_journals(summary["top_journals"])
                st.pyplot(fig_journals)
                st.table(summary["top_journals"].reset_index().rename(columns={"index":"journal", journal_col if journal_col else 0:"count"}).head(10))

            # Abstract wordcount histogram
            if "abstract_word_count" in df_clean.columns:
                st.subheader("Abstract Word Count Distribution")
                fig_hist = plot_abstract_wordcount_hist(df_clean)
                st.pyplot(fig_hist)

            # Scatter plot
            if "abstract_word_count" in df_clean.columns and "title" in df_clean.columns:
                st.subheader("Title length vs Abstract word count")
                fig_scatter = plot_titlelen_vs_abstractwords(df_clean)
                st.pyplot(fig_scatter)

            # Top title words
            if "top_title_words" in summary:
                st.subheader("Top words in titles (sample)")
                top_words = summary["top_title_words"][:30]
                st.write(pd.DataFrame(top_words, columns=["word","count"]).head(20))

            # Show sample rows (interactive)
            st.subheader("Explore rows")
            num_show = st.slider("Rows to show", 5, 200, 10)
            st.dataframe(df_clean.head(num_show))

            st.success("Analysis complete. You can download figures by right-clicking them in the browser or adapt the code to save to disk.")
        except Exception as e:
            st.error(f"Failed to load or analyze data: {e}")

st.markdown("---")
st.markdown("**Notes:** This is a simple demo app. For larger CORD-19 metadata files, load a sample (few thousand rows) or run the script locally (not in the browser) to avoid running out of memory.")
