"""
===============================================================================
STARTER TEMPLATE: from a CSV file to your first model
===============================================================================
A beginner's guide to processing a new dataset, one step at a time.
It follows the same order as the README:

    PART 1  Load and look at the data           (always)
    PART 2  Clean the data                      (always)
    PART 3A Classification: predict a category  (if your data has a target column)
    PART 3B Clustering: find groups             (if it has no target column)

How to use it:
    1. Put your CSV file in the same folder as this script.
    2. Change the SETTINGS section below. It's the only part you have to edit.
    3. Right-click this file -> Run. Read the output from top to bottom.
    4. Screenshot the outputs and charts for your report.

This file deliberately stays simple. When you need more, look at the
weekly tutorials:
    Week2.py  more classifiers and ensembles (AdaBoost, bagging, voting)
    Week3.py  more clustering (Fuzzy C-Means, DBSCAN, hierarchical)
    Week4.py  shopping baskets (association rules)
    Week6.py  text data (reviews, tweets)

Packages needed (type in the PyCharm Terminal):
    python -m pip install numpy pandas matplotlib scikit-learn
===============================================================================
"""

import numpy as np                  # maths and arrays
import pandas as pd                 # tables (DataFrames)
import matplotlib.pyplot as plt     # charts

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, silhouette_score)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# =============================================================================
# SETTINGS: change these for your own dataset
# =============================================================================

# The name of your data file (it must be in the same folder as this script).
FILE_NAME = "Practice_Data.csv"

# What separates the columns in the file?
#   ","  = normal CSV (most files)
#   "\t" = tab (Practice_Data.csv uses tabs)
#   ";"  = semicolon (common in European files)
# Tip: if everything ends up in ONE column, the separator is wrong.
SEPARATOR = "\t"

# The column you want to predict (the "answer"), e.g. "Churn" or "price_range".
# Set it to None if there is no answer column and you want to find groups
# (clustering) instead.
TARGET_COLUMN = "price_range"

# Columns that won't help the model: ID numbers, names, free text, etc.
COLUMNS_TO_DROP = []

# Clustering only: how many groups to make. Run once, look at the elbow and
# silhouette charts, then come back and set the best number here.
NUMBER_OF_CLUSTERS = 4

# Example settings for the Week 3 mall data (clustering):
#   FILE_NAME = "Mall_Customers.csv"
#   SEPARATOR = ","
#   TARGET_COLUMN = None
#   COLUMNS_TO_DROP = ["CustomerID", "Gender"]
#   NUMBER_OF_CLUSTERS = 5


# =============================================================================
# PART 1: LOAD AND LOOK AT THE DATA
# =============================================================================

# -----------------------------------------------------------------------------
# Step 1: Load the file into a table called df ("data frame")
# -----------------------------------------------------------------------------
# If you get a UnicodeDecodeError, add  encoding="latin1"  inside the brackets.
# For an Excel file use:  df = pd.read_excel("file.xlsx")
df = pd.read_csv(FILE_NAME, sep=SEPARATOR)

print("STEP 1: data loaded")
print("Rows and columns:", df.shape)

# -----------------------------------------------------------------------------
# Step 2: Look at the data BEFORE changing anything
# -----------------------------------------------------------------------------
# Screenshot these for the report: they show you checked the data.
print("\nSTEP 2: first 5 rows")
print(df.head())

print("\nColumn names, types and non-missing counts")
df.info()       # df.info() prints by itself, so it doesn't need print()

print("\nSummary statistics: look for strange min/max values (e.g. age 999)")
print(df.describe())

print("\nMissing values in each column")
print(df.isnull().sum())

print("\nNumber of duplicate rows:", df.duplicated().sum())


# =============================================================================
# PART 2: CLEAN THE DATA
# For the report, write down WHAT you did at each step and WHY.
# =============================================================================

# -----------------------------------------------------------------------------
# Step 3: Remove duplicate rows (exact copies of another row)
# -----------------------------------------------------------------------------
rows_before = len(df)
df = df.drop_duplicates()
print("\nSTEP 3: removed", rows_before - len(df), "duplicate rows")

# -----------------------------------------------------------------------------
# Step 4: Drop columns that don't help
# -----------------------------------------------------------------------------
df = df.drop(columns=COLUMNS_TO_DROP)
print("STEP 4: dropped columns", COLUMNS_TO_DROP)

# -----------------------------------------------------------------------------
# Step 5: Fix wrong values and messy text (OPTIONAL: depends on your data)
# -----------------------------------------------------------------------------
# Remove the # at the start of a line to use an example.
#
# A number stored as text, e.g. "£5.99" -> 5.99:
#   df["Price"] = pd.to_numeric(df["Price"].str.replace("£", ""), errors="coerce")
#
# Messy categories, e.g. " Male", "male", "MALE" -> "male":
#   df["Gender"] = df["Gender"].str.strip().str.lower()
#
# An impossible zero that really means "unknown". For example, in
# Practice_Data.csv some phones have a screen width (sc_w) of 0:
#   df["sc_w"] = df["sc_w"].replace(0, np.nan)    # np.nan = missing
print("STEP 5: (optional fixes: see the examples in the code)")

# -----------------------------------------------------------------------------
# Step 6: Fill in missing values
# -----------------------------------------------------------------------------
# Rows with no answer can't be used for training, so delete them.
if TARGET_COLUMN is not None:
    df = df.dropna(subset=[TARGET_COLUMN])

# For every other column with gaps:
#   number column -> fill with the MEDIAN (the middle value; outliers don't affect it)
#   text column   -> fill with the MODE (the most common value)
for column in df.columns:
    if df[column].isnull().sum() > 0:
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].median())
            print("STEP 6: filled gaps in", column, "with the median")
        else:
            df[column] = df[column].fillna(df[column].mode()[0])
            print("STEP 6: filled gaps in", column, "with the most common value")
print("STEP 6: missing values left:", df.isnull().sum().sum())

# -----------------------------------------------------------------------------
# Step 7: Remove outliers (OPTIONAL)
# -----------------------------------------------------------------------------
# Only do this if describe() showed impossible values. The IQR rule keeps values
# between Q1 - 1.5*IQR and Q3 + 1.5*IQR. Replace "Income" with your column:
#   Q1 = df["Income"].quantile(0.25)
#   Q3 = df["Income"].quantile(0.75)
#   IQR = Q3 - Q1
#   df = df[(df["Income"] >= Q1 - 1.5 * IQR) & (df["Income"] <= Q3 + 1.5 * IQR)]
print("STEP 7: (optional outlier removal: see the example in the code)")

print("\nClean data: rows and columns =", df.shape)


# =============================================================================
# PART 3A: CLASSIFICATION (predict the TARGET_COLUMN)
# Used when TARGET_COLUMN has a name. See Week2.py for more models.
# =============================================================================
if TARGET_COLUMN is not None:

    # -------------------------------------------------------------------------
    # Step 8: Split the inputs (X) from the answer (y)
    # -------------------------------------------------------------------------
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Models only understand numbers.
    # The answer: turn text labels (e.g. "Yes"/"No") into 0/1 with a LabelEncoder.
    if not pd.api.types.is_numeric_dtype(y):
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
        print("\nSTEP 8: answer labels turned into numbers:",
              dict(zip(encoder.classes_, range(len(encoder.classes_)))))

    # The inputs: one-hot encoding makes a 0/1 column for each category
    # (e.g. City -> City_Leeds, City_York). Number columns are left as they are.
    X = pd.get_dummies(X, drop_first=True, dtype=int)
    print("STEP 8: number of input columns:", X.shape[1])

    # -------------------------------------------------------------------------
    # Step 9: Check the class balance
    # -------------------------------------------------------------------------
    # If one answer is much more common (e.g. 90% "No"), say so in the report
    # and trust the F1 score more than accuracy.
    print("\nSTEP 9: how many of each answer")
    print(pd.Series(y).value_counts())

    # -------------------------------------------------------------------------
    # Step 10: Train/test split
    # -------------------------------------------------------------------------
    # The models learn from 80% of the rows and are tested on the other 20%,
    # which they have never seen. That gives a fair score.
    # stratify=y keeps the same mix of answers in both parts.
    # random_state=42 gives the same split every time you run the file.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print("\nSTEP 10: training rows:", len(X_train), "| test rows:", len(X_test))

    # -------------------------------------------------------------------------
    # Step 11: Scale the numbers
    # -------------------------------------------------------------------------
    # Puts every column on the same scale (average 0, spread 1), so a column
    # with big numbers (e.g. salary) doesn't drown out a small one (e.g. age).
    # Needed by KNN and logistic regression; trees don't mind either way.
    # IMPORTANT: fit the scaler on the TRAINING data only, then reuse it on the
    # test data. Fitting on the test data is "data leakage".
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("STEP 11: data scaled")

    # -------------------------------------------------------------------------
    # Step 12: Train several models and compare them
    # -------------------------------------------------------------------------
    # Each model follows the same 3 lines: create -> fit (learn) -> predict.
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    print("\nSTEP 12: model results on the test data")
    results = []
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)          # learn from the training rows
        y_pred = model.predict(X_test_scaled)       # predict the unseen test rows

        accuracy = accuracy_score(y_test, y_pred)   # % of answers correct
        # F1 balances precision and recall. average="macro" gives every class
        # equal weight, so it also works with more than 2 classes.
        f1 = f1_score(y_test, y_pred, average="macro")

        results.append({"Model": name, "Accuracy": round(accuracy, 3), "F1": round(f1, 3)})
        print(f"  {name:<20} accuracy = {accuracy:.3f}   F1 = {f1:.3f}")

    results = pd.DataFrame(results).sort_values("F1", ascending=False)
    best_name = results.iloc[0]["Model"]
    best_model = models[best_name]
    print("\nBest model:", best_name)

    # Bar chart of the scores: a good screenshot for the report
    results.plot(x="Model", y=["Accuracy", "F1"], kind="bar", figsize=(8, 4), rot=0)
    plt.title("Model comparison on the test data")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # Step 13: Look more closely at the best model
    # -------------------------------------------------------------------------
    best_pred = best_model.predict(X_test_scaled)

    # Precision, recall and F1 for each class
    print("\nSTEP 13: classification report for", best_name)
    print(classification_report(y_test, best_pred))

    # Confusion matrix: rows = the real answer, columns = the prediction.
    # The diagonal holds the correct answers; everything else is a mistake.
    print("Confusion matrix:")
    print(confusion_matrix(y_test, best_pred))

    # -------------------------------------------------------------------------
    # Step 14: Which columns matter most? (from the random forest)
    # -------------------------------------------------------------------------
    # Useful for business recommendations: "RAM drives the price most".
    importance = pd.Series(models["Random Forest"].feature_importances_, index=X.columns)
    top10 = importance.sort_values(ascending=False).head(10)
    print("\nSTEP 14: the 10 most important columns (random forest)")
    print(top10.round(3))

    top10.sort_values().plot(kind="barh", figsize=(7, 4))
    plt.title("Most important columns (random forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()


# =============================================================================
# PART 3B: CLUSTERING (find groups when there is no answer column)
# Used when TARGET_COLUMN = None. See Week3.py for more methods.
# =============================================================================
else:

    # -------------------------------------------------------------------------
    # Step 8: Prepare the inputs
    # -------------------------------------------------------------------------
    # Clustering has no answer column, so every remaining column is an input.
    # Text columns become 0/1 columns (one-hot encoding).
    X = pd.get_dummies(df, drop_first=True, dtype=int)
    print("\nSTEP 8: columns used for clustering:", list(X.columns))

    # -------------------------------------------------------------------------
    # Step 9: Scale the numbers (ALWAYS do this for clustering)
    # -------------------------------------------------------------------------
    # Clustering measures distances between points. Without scaling, a column
    # with big numbers (e.g. income) would decide everything.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("STEP 9: data scaled")

    # -------------------------------------------------------------------------
    # Step 10: Choose the number of groups (k)
    # -------------------------------------------------------------------------
    # Try k = 2 to 10 and measure:
    #   inertia    = how spread out each group is (lower is better, but it
    #                always falls; look for the "elbow" where it levels off)
    #   silhouette = how well the groups are separated (-1 to 1, higher is better)
    k_values = range(2, 11)
    inertias = []
    silhouettes = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
        print(f"  k = {k:2d}   inertia = {kmeans.inertia_:9.1f}   silhouette = {silhouettes[-1]:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(list(k_values), inertias, marker="o")
    axes[0].set_title("Elbow plot")
    axes[0].set_xlabel("Number of groups (k)")
    axes[0].set_ylabel("Inertia")
    axes[1].plot(list(k_values), silhouettes, marker="o", color="green")
    axes[1].set_title("Silhouette score")
    axes[1].set_xlabel("Number of groups (k)")
    axes[1].set_ylabel("Silhouette")
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # Step 11: Run K-Means with the chosen k
    # -------------------------------------------------------------------------
    kmeans = KMeans(n_clusters=NUMBER_OF_CLUSTERS, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, df["Cluster"])
    print(f"\nSTEP 11: K-Means with k = {NUMBER_OF_CLUSTERS}, silhouette = {score:.3f}")

    # -------------------------------------------------------------------------
    # Step 12: Describe each group (the most useful output for a business)
    # -------------------------------------------------------------------------
    # The averages are in the ORIGINAL units (not scaled), so you can name each
    # group, e.g. "young, low income, high spending".
    print("\nSTEP 12: how many in each group")
    print(df["Cluster"].value_counts().sort_index())
    print("\nAverage of each column per group")
    print(X.assign(Cluster=df["Cluster"]).groupby("Cluster").mean().round(1))

    # -------------------------------------------------------------------------
    # Step 13: Draw the groups
    # -------------------------------------------------------------------------
    # PCA squashes all the columns into 2, so the groups can be drawn on a flat
    # chart. The 2 new axes don't mean anything on their own.
    points_2d = PCA(n_components=2).fit_transform(X_scaled)
    plt.figure(figsize=(6, 5))
    plt.scatter(points_2d[:, 0], points_2d[:, 1], c=df["Cluster"], cmap="tab10", s=25)
    plt.title(f"K-Means groups (k = {NUMBER_OF_CLUSTERS}), drawn with PCA")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.show()

    # To plot two of your own columns instead, e.g. income against spending:
    #   plt.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"], c=df["Cluster"], cmap="tab10")
    #   plt.show()


# =============================================================================
# WHAT NEXT?
# =============================================================================
# Report checklist (README section 6):
#   [ ] Business problem: what does the company want to know?
#   [ ] Data issues found (missing values, duplicates, outliers, imbalance)
#       and how you fixed them, and WHY
#   [ ] The techniques you used and why they suit this data
#   [ ] Results (accuracy / F1 or silhouette) with screenshots
#   [ ] A business recommendation
#   [ ] Limitations, and what you'd try with more time
#
# Ideas to go further:
#   - Classification: try ensembles from Week2.py (AdaBoost, bagging, voting),
#     or cross_val_score for a fairer average score over several splits.
#   - Clustering: compare K-Means with DBSCAN or hierarchical clustering (Week3.py).
#   - A number to predict (e.g. price)? Swap the classifiers for
#     LinearRegression / RandomForestRegressor and score with r2_score.
print("\nFinished. Now write up what you found (see the checklist at the end of the code).")
