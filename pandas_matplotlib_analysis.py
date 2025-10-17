# Analyzing and visualizing data using pandas and matplotlib taught in project plan program 
'''objective is to :
-load,explore and analyze data using pandas library 
-visualize this data for insights and seaborn
'''


# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Enable inline plotting (for Jupyter)
%matplotlib inline

# Task 1: Load and Explore the Dataset
try:
    # Load the Iris dataset from seaborn’s built-in datasets
    df = sns.load_dataset("iris")
    print("✅ Dataset loaded successfully!\n")
except FileNotFoundError:
    print("❌ File not found. Please check the dataset path.")
except Exception as e:
    print(f"⚠️ Error loading dataset: {e}")
else:
    # Display first few rows
    print("🔹 First five rows of the dataset:")
    print(df.head())

    # Display info
    print("\n🔹 Dataset Info:")
    print(df.info())

    # Check for missing values
    print("\n🔹 Missing Values:")
    print(df.isnull().sum())

    # Clean dataset: (Iris dataset has no missing values, but let's include handling)
    df = df.dropna()
    print("\n✅ Cleaned dataset (no missing values remaining).")

# Task 2: Basic Data Analysis

# Descriptive statistics
print("\n📊 Basic Statistics:")
print(df.describe())

# Group by species and compute mean of numeric columns
grouped_means = df.groupby("species").mean(numeric_only=True)
print("\n📈 Mean values by Species:")
print(grouped_means)

# Identify simple pattern
print("\n💡 Observation:")
print("Setosa flowers generally have smaller petal and sepal sizes than Virginica or Versicolor.")

# Task 3: Data Visualization using seaborn

# Set Seaborn style
sns.set(style="whitegrid")

# 1️⃣ Line Chart - showing sepal length per sample
plt.figure(figsize=(8, 4))
plt.plot(df.index, df["sepal_length"], label="Sepal Length", color='blue')
plt.title("Sepal Length Trend Across Samples")
plt.xlabel("Sample Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.show()

# 2️⃣ Bar Chart - average petal length per species
plt.figure(figsize=(6, 4))
sns.barplot(x="species", y="petal_length", data=df, palette="viridis")
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# 3️⃣ Histogram - distribution of sepal width
plt.figure(figsize=(6, 4))
plt.hist(df["sepal_width"], bins=15, color="skyblue", edgecolor="black")
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# 4️⃣ Scatter Plot - sepal length vs petal length
plt.figure(figsize=(6, 4))
sns.scatterplot(x="sepal_length", y="petal_length", hue="species", data=df, palette="Set2")
plt.title("Sepal Length vs Petal Length by Species")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()

# ------------------------------------------------------------
# Final Observations
# ------------------------------------------------------------
print("\n📘 Findings & Observations:")
print("- Setosa species generally have the smallest petals and sepals.")
print("- Virginica species tend to have the largest petal length and sepal width.")
print("- There’s a clear positive correlation between sepal length and petal length across all species.")
print("- Visualizations confirm how measurements vary by flower species.")
