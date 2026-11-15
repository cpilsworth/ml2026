# A Guide to Choosing the Right Analysis and Model

A guide to understanding the different types of analysis, when to use each one,
and how to go from "here is a data file" to "this is the model I chose, and why".

It builds on the `README` (installing packages, cleaning code, a simple guide to each
model) and `Starter_Template.py` (a runnable first pass at any dataset). The worked
examples use the files in this folder, and the numbers quoted come from running them.

To keep things simple, it only uses the models taught in the tutorials (Weeks 2, 3,
4 and 6), plus the two regression models the README mentions. Section 5 lists them all.

---

## Contents

1. [Start with the question, not the model](#1-start-with-the-question-not-the-model)
2. [The main types of analysis](#2-the-main-types-of-analysis)
3. [Reading a data file: what kind of problem is it?](#3-reading-a-data-file-what-kind-of-problem-is-it)
4. [The decision flowchart](#4-the-decision-flowchart)
5. [Choosing a model within each type](#5-choosing-a-model-within-each-type)
6. [How to choose: a repeatable process](#6-how-to-choose-a-repeatable-process)
7. [Choosing the right score](#7-choosing-the-right-score)
8. [Common traps](#8-common-traps)
9. [Worked examples with the files in this folder](#9-worked-examples-with-the-files-in-this-folder)
10. [Picking two techniques (exam)](#10-picking-two-techniques-exam)
11. [One-page checklist](#11-one-page-checklist)

---

## 1. Start with the question, not the model

The most common mistake is to start with "I'll use a random forest". The model is
the **last** decision, not the first. The order is:

```
1. What does the business want to know or do?        (the question)
2. What does the data file contain?                  (rows, columns, an answer column?)
3. Which TYPE of analysis answers that question      (classification, clustering, ...)
   with that data?
4. Which MODELS of that type suit this data?         (size, columns, explainability)
5. How will I judge them?                            (accuracy, F1, silhouette, lift, ...)
6. What does the result mean for the business?       (the recommendation)
```

The same file can support different analyses depending on the question. Take
`Mall_Customers.csv`:

| Question | Type of analysis |
|---|---|
| "What kinds of customers do we have?" | Clustering (find segments) |
| "Can we predict a customer's spending score from age and income?" | Regression (predict a number) |
| "Is this customer a high spender, yes or no?" | Classification (once you define "high") |

So always write the business question down first. It decides everything else.

---

## 2. The main types of analysis

Machine learning splits into two big families, plus some special cases for
particular kinds of data.

### Supervised learning: there is an answer column

You have past examples **with the answer**, and you want to predict the answer for
new rows. The answer column is called the **target**, **label** or **y**. The
other columns are the **features**, **inputs** or **X**.

| Type | The answer is... | Example questions | Course |
|---|---|---|---|
| **Classification** | a category: Yes/No, A/B/C, band 0–3 | Will this customer leave? Is this tumour malignant? Which price band is this phone in? | Week 2 |
| **Regression** | a number on a scale | What will this house sell for? How much will this customer spend next month? | README (not in a tutorial) |

The model learns the link between inputs and answers from the **training** rows,
and is tested on **test** rows it has never seen.

### Unsupervised learning: there is no answer column

There is nothing to predict. Instead you look for **structure** in the data.

| Type | What it finds | Example questions | Course |
|---|---|---|---|
| **Clustering** | groups of similar rows | What customer segments do we have? Which stores behave alike? | Week 3 |
| **Association rules** | items that occur together | What do people buy together? Which products should we bundle? | Week 4 |

Unsupervised results have no "right answer" to check against. You judge them with
internal scores (silhouette, lift) **and** by whether they make business sense.

### Special data types

| Data | What changes | Course |
|---|---|---|
| **Text** (reviews, tweets, emails) | The text must be cleaned and turned into numbers (word counts or TF-IDF) first. After that it's usually **classification**, e.g. positive/negative. | Week 6 |
| **Transactions** (one row per item bought) | The rows must be turned into one row per basket before association rules can run. | Week 4 |
| **Dates over time** (daily sales, stock prices) | This is **time-series forecasting**, which isn't covered in these tutorials. Don't shuffle time data into a random train/test split. | Not covered |

### Always first: exploratory analysis

Whatever the type, you **always** start by exploring: shape, column types,
`describe()`, missing values, duplicates, class balance and charts. This is Part 1 of
`Starter_Template.py`. Exploring tells you which type of analysis is possible, and it
is where you find the data issues to write about in the report.

---

## 3. Reading a data file: what kind of problem is it?

Open the file (run Part 1 of `Starter_Template.py`) and answer these questions in order.

### Q1. What does one row represent?

| One row is... | Usually means |
|---|---|
| One person, customer, patient or product, with a column describing an outcome | Classification or regression |
| One person, customer or product, with **no** outcome column | Clustering |
| One **item** within a purchase (the same invoice number repeats over several rows) | Association rules (build baskets first) |
| One piece of text plus a label | Text classification (NLP) |
| One point in time (a date column, in order) | Time series (not covered) |

### Q2. Is there an answer column?

Look for a column that is the **result** of everything else, often with a name
like `Churn`, `Outcome`, `Label`, `Class`, `Target`, `Default`, `Survived`,
`sentiment`, `price_range` or `Diagnosis`. Ask yourself: "Would the business want to
know this **before** it happens?"

- **Yes:** supervised learning. Go to Q3.
- **No:** unsupervised learning (clustering or association rules).

### Q3. What kind of values does the answer column hold?

| Answer values | Type |
|---|---|
| Words: Yes/No, positive/negative, A/B/C | Classification |
| A few whole numbers that stand for groups, e.g. `price_range` 0–3, rating 1–5 | Classification (the numbers are labels, not amounts) |
| Many different numbers on a scale, e.g. price, salary, temperature | Regression |

Quick check in Python: `df["column"].nunique()`. A handful of values usually means
classification; hundreds of values means regression.

### Q4. What are the other columns like?

| What you see | What it means for you |
|---|---|
| ID numbers, names, phone numbers | Drop them. They identify a row but don't explain anything. |
| Text categories (City, Gender) | Encode them (one-hot) before modelling. |
| Long free text | NLP (Week 6). |
| Numbers on very different scales (age 18–70, income 15,000–137,000) | Scale them for KNN, SVM, logistic regression and all clustering. |
| A column that gives the answer away (e.g. `cancellation_date` when predicting churn) | Remove it. This is "target leakage" and makes scores look unrealistically good. |
| Impossible values (age 999, screen width 0) | Clean them (Step 5/7 of the template). |

### Q5. How big is it, and is it balanced?

- **Rows:** a few hundred rows calls for simple models and cross-validation. Hundreds of thousands calls for fast models (see section 5).
- **Columns:** many columns, especially irrelevant ones, hurt some models (KNN in particular).
- **Balance:** `df["target"].value_counts()`. If one class makes up 80% or more, accuracy is misleading; use F1 (section 7).

---

## 4. The decision flowchart

```
                        What is the business question?
                                     |
       +-----------------------------+------------------------------+
       |                             |                              |
 "Predict something            "What groups or           "What happens/is
  for each row"                  segments exist?"           bought together?"
       |                             |                              |
 Is there a column              No answer column           Rows are items within
 holding the answer?                 |                     transactions/baskets
       |                             v                              |
   +---+-------------+         CLUSTERING (Week 3)                  v
   |                 |         KMeans, DBSCAN, ...       ASSOCIATION RULES (Week 4)
 Categories       A number                                   apriori -> rules
 (Yes/No, 0-3)    on a scale
   |                 |
   v                 v
 CLASSIFICATION   REGRESSION
 (Week 2)         (README)

 Is the main information free TEXT?  ->  clean + vectorise (Week 6),
                                          then usually CLASSIFICATION
 Are the rows dates in order?        ->  time series (not covered: ask)
```

---

## 5. Choosing a model within each type

This guide only uses the models from the tutorials, plus the two regression models
the README mentions. Here they all are in one place:

| Type | Models | Where to find the code |
|---|---|---|
| Classification | Decision Tree, Random Forest, KNN, SVM, AdaBoost, Bagging, Voting (soft and hard) | `Week2.py` (Decision Tree, Random Forest and KNN also in `Starter_Template.py`) |
| Classification | Logistic Regression | `Week6.py` (text) and `Starter_Template.py` (tables) |
| Text classification | Naive Bayes, Logistic Regression, SGD Classifier | `Week6.py` |
| Clustering | K-Means, Fuzzy C-Means, DBSCAN, Agglomerative | `Week3.py` (K-Means also in `Starter_Template.py`) |
| Association rules | Apriori | `Week4.py` |
| Regression | Linear Regression, Random Forest Regressor | README only (no tutorial code) |

Once you know the **type**, pick 2–4 of these and compare them. The tables below show
what tips the choice one way or the other.

### 5.1 Classification

| Model | Choose it when... | Avoid or be careful when... | Scale? | Week |
|---|---|---|---|---|
| **Logistic Regression** | You want a fast, explainable starting model that gives probabilities ("80% chance of churn"). Works well when the answer depends roughly on straight-line trends. | The relationships are very curved or complicated. | Yes | 6 |
| **Decision Tree** | You must **explain** the model to non-experts (you can draw it). | You leave it unlimited: it memorises the training data (overfits). Set `max_depth`. | No | 2 |
| **Random Forest** | You want a strong, reliable default that also tells you which columns matter. Copes with messy data and outliers. | You need a single, simple explanation. | No | 2 |
| **KNN** | The data is small and every column is relevant. | There are many irrelevant columns (distances become noise) or lots of rows (slow). | Yes | 2 |
| **SVM** | Medium-sized data, and accuracy matters more than explanation. | The data is large (slow) or you need to explain the model. | Yes | 2 |
| **AdaBoost** | You want high accuracy from many small trees, each fixing the previous one's mistakes. | The data is noisy or has many outliers. | No | 2 |
| **Bagging** | One model is unstable (its score jumps around between folds). Bagging trains it many times on random samples and takes a vote. | You need to explain it, or need speed. | Depends | 2 |
| **Voting (soft or hard)** | Several different models each do well; let them vote. | You need to explain it, or need speed. | Depends | 2 |

### 5.2 Regression (README only)

The tutorials don't include regression, but the README lists two models in case a
problem asks you to predict a number:

| Model | Choose it when... | Be careful when... |
|---|---|---|
| **Linear Regression** | You want a simple, explainable model ("each extra bedroom adds £15k"). | The relationships are curved; outliers pull the line. |
| **Random Forest Regressor** | You want more accuracy and the relationships are complicated. | It's harder to explain. |

They work like the classifiers: `fit`, then `predict`, then score with `r2_score` and
`mean_absolute_error` instead of accuracy.

### 5.3 Clustering

| Model | Choose it when... | Be careful when... |
|---|---|---|
| **K-Means** | The default. Groups are roughly round blobs of similar size, and you can choose k. | Odd-shaped groups or strong outliers. You must pick k. |
| **Fuzzy C-Means** | People realistically belong partly to several groups ("70% bargain hunter"). | The package can be awkward to install. |
| **DBSCAN** | Groups have odd shapes, you don't know how many there are, or you want outliers flagged as noise (`-1`). | Choosing `eps` and `min_samples` is fiddly. |
| **Agglomerative (hierarchical)** | Small data, and you want the same result on every run (no random starting points). | Large data (slow). |

**Choosing k:** look at the elbow plot and the silhouette score (both in Week 3),
**and** ask how many segments the business can actually act on. Five clear segments
beat nine confusing ones.

### 5.4 Association rules

There's only one model, **apriori** (Week 4). The real choices are its **settings**:

| Setting | Start at | Adjust |
|---|---|---|
| `min_support` | 0.05 (5% of baskets) | Lower it (0.02, 0.01) if you get few or no rules. Raise it if you get thousands. |
| `metric` / `min_threshold` | `lift`, 1.0 | Only keep rules with lift above 1 (a real link, not chance). |
| Subset | One country (Week 4 uses France) | Baskets from very different markets dilute each other. |

### 5.5 Text

| Approach | Notes |
|---|---|
| **Naive Bayes + word counts (`CountVectorizer`)** | The classic fast starting model. Always try it first. |
| **Logistic Regression or SGD + TF-IDF** | Usually the most accurate of the three, and the word weights explain the result. |
| **Word pairs (`ngram_range=(1, 2)`)** | Captures "not good". Make sure cleaning doesn't remove "not"! |

### 5.6 What the data itself tells you

| If your data has... | Lean towards... | Lean away from... |
|---|---|---|
| Few rows (under about 1,000) | Simple models, with cross-validation | — |
| Very many rows (100k+) | Logistic Regression, Naive Bayes / SGD for text, Random Forest | SVM, KNN, Agglomerative |
| Many columns, several irrelevant | Random Forest | KNN |
| A need for explanation | Decision Tree, Logistic Regression | SVM, AdaBoost, Bagging, Voting |
| Strong outliers | Decision Tree, Random Forest; DBSCAN for clustering | Logistic Regression, AdaBoost, K-Means (or remove the outliers) |
| Imbalanced classes | Stratified split, F1 score | Judging on accuracy alone |
| One very strong straight-line relationship | Logistic Regression (it may beat complex models) | Assuming complex is always better |

---

## 6. How to choose: a repeatable process

Use the same process every time. It is also a good structure for the methods section
of a report.

### Step 1: Set a baseline to beat

Before any real model, ask: **"What score would a guess get?"**

- **Classification:** always predicting the most common class. On `Practice_Data.csv`, with 4 equal classes, that is **25%**. On the breast cancer data (357 benign, 212 malignant) it is **63%**.
- **Regression:** always predicting the average (R² = 0).

A model is only useful if it clearly beats the baseline. "90% accuracy" means nothing
if 90% of the rows are "No".

### Step 2: Try a simple, explainable model

Logistic regression or a shallow decision tree. Sometimes it is the winner (see the
phone example in section 9).

### Step 3: Try 1–3 stronger models

Random Forest, SVM, AdaBoost or a voting ensemble (Week 2), whichever suits the data (section 5).

### Step 4: Compare them fairly

- Same train/test split for every model (`random_state=42`).
- Same preparation (scaling fitted on the training data only).
- Same score (section 7).
- Better still, use **cross-validation** (`cross_val_score`). It averages the score over several splits, so one lucky split can't mislead you. The **spread** (standard deviation) shows how stable the model is.

### Step 5: Tune a little (optional)

Each model has a few important settings ("hyperparameters"): `max_depth` for trees,
`n_neighbors` for KNN, `n_estimators` for forests and AdaBoost. Week 2 shows the simple
way to tune them: build the same model with a few different values (KNN with k = 1, 3,
5, 7, 9; SVM with degree 1 to 5), score each with `cross_val_score`, and compare the
results in a boxplot.

### Step 6: Decide and justify

- Pick the best score **unless** the difference is tiny. Then prefer the simpler, more explainable model.
- Check the **test** data once, at the end. Don't keep tweaking the model to improve the test score; that turns the test set into training data.
- Write it in the README's format: *"I chose [model] because [strength]. Its weakness is [weakness], so I [what you did about it]."*

---

## 7. Choosing the right score

### Classification

| Score | What it means | Use it when |
|---|---|---|
| **Accuracy** | % of predictions that are correct | The classes are roughly balanced |
| **Precision** | Of the rows predicted "Yes", how many really were "Yes"? | False alarms are costly (e.g. marking real email as spam) |
| **Recall** | Of the real "Yes" rows, how many did we catch? | Misses are costly (e.g. missing a cancer, missing fraud) |
| **F1** | A balance of precision and recall | The classes are imbalanced, or you need one number |
| **Confusion matrix** | A table of exactly which classes get confused | Always: it shows *where* the model goes wrong |

**Ask what a mistake costs.** In the breast cancer data, telling a patient with a
malignant tumour that it is benign (a false negative) is far worse than the reverse.
So **recall for "malignant"** is the number that matters most.

### Regression

| Score | Meaning |
|---|---|
| **R²** | Share of the variation explained (1 = perfect, 0 = no better than the average) |
| **MAE** | Average size of the error, in the target's units ("off by £12k on average") |

### Clustering

| Score | Meaning |
|---|---|
| **Inertia (elbow plot)** | How tight the groups are. It always falls as k rises; look for the bend. |
| **Silhouette** | −1 to 1: how well separated the groups are (higher is better) |
| **Business sense** | Can you name each group from its averages, and act on it? |

### Association rules

| Score | Meaning |
|---|---|
| **Support** | How often the items appear together (is it common enough to matter?) |
| **Confidence** | If someone buys A, how likely are they to buy B? |
| **Lift** | How much more often than chance (> 1 = a real link) |

---

## 8. Common traps

| Trap | What goes wrong | How to avoid it |
|---|---|---|
| **Data leakage** | Information from the test set leaks into training (e.g. fitting the scaler on all the data), so scores look better than reality. | Split first, then `fit` the scaler or vectoriser on the training data and only `transform` the test data. |
| **Target leakage** | A column that gives away the answer (e.g. "refund issued" when predicting complaints). | Ask of each column: "Would I know this **before** the outcome?" |
| **Overfitting** | The model memorises the training data and does worse on new data. | Compare the training score with the test score. A big gap means overfitting. Limit `max_depth`, use cross-validation. |
| **Judging on accuracy alone with imbalanced data** | 95% accuracy by always saying "No". | Check the balance, use F1/recall, `stratify=y`. |
| **Not scaling** | KNN, SVM and K-Means are dominated by the column with the biggest numbers. | `StandardScaler` (not needed for trees). |
| **Clustering on IDs** | The CustomerID column becomes the main "feature". | Drop IDs before clustering. |
| **Treating number codes as amounts** | `price_range` 0–3 is a category, not a quantity. | Check `nunique()` and what the numbers mean. |
| **Reading rules as causes** | "Bread → butter" doesn't mean bread *causes* butter sales. | Say "bought together", not "causes". |

---

## 9. Worked examples with the files in this folder

### Example A: `Practice_Data.csv`, mobile phone price bands

**Looking at the file**
- 2,000 rows, one per phone, with 20 spec columns (RAM, battery, pixels, camera, 3G/4G...).
- The columns are separated by **tabs** (`sep="\t"`).
- Answer column: `price_range`, values 0, 1, 2, 3. Only 4 values, so these are **categories** (price bands), not prices.
- Balance: exactly 500 of each band, so it is balanced. Accuracy is fine, and the baseline is 25%.
- Data issues: no missing values or duplicates, but `sc_w` (screen width) is 0 for 180 phones, which is impossible, so treat it as missing. `ram` has a correlation of 0.92 with `price_range`; nothing else is above 0.21.

**Type of analysis:** Classification, because there is an answer column holding categories.

**Results** (`Starter_Template.py` or the notebook, 80/20 split; numbers can differ
slightly between library versions):

| Model | Test accuracy |
|---|---|
| Logistic Regression | **0.965** |
| Random Forest | 0.877 |
| Decision Tree (depth 5) | 0.830 |
| KNN (all 20 columns) | 0.500 |

**What we learn about model choice**
- **The simplest model won.** The price band depends mostly on RAM, battery and screen pixels in a fairly straight-line way, which is exactly what logistic regression models. Always include a simple model; don't assume complex is better.
- **KNN failed because of irrelevant columns.** KNN measures distance across all 20 scaled columns equally, and 16 of them barely matter, so "nearest neighbours" are mostly decided by noise. Using only `ram`, `battery_power`, `px_width` and `px_height`, KNN jumps from **0.50 to 0.90**. This is what "avoid KNN with many irrelevant columns" means in practice.
- **Overfitting:** a decision tree with no depth limit scores **100%** on the training data but **83%** on the test data. It memorised the training rows.
- **Business finding:** RAM is by far the biggest driver of price band.

### Example B: `Mall_Customers.csv`, customer segments

**Looking at the file**
- 200 rows, one per customer: CustomerID, Gender, Age, Annual Income (k$), Spending Score (1–100).
- No outcome to predict, and the business wants to know "what types of customer do we have?", so this is **clustering**.
- Drop `CustomerID` (an identifier). Scale the columns, because income and spending score are on different scales.

**Choosing k**

| k | 4 | 5 | 6 |
|---|---|---|---|
| Silhouette | 0.404 | 0.417 | **0.428** |

The silhouette score is highest at k=6, but 5 is close and the elbow bends around 4–6.
Week 3 chose **k=5**, which is a reasonable judgment because five segments are easier
to act on. That is the point: the score guides you, and business sense decides.

**The five K-Means segments (averages in original units)**

| Segment | Size | Age | Income (k$) | Spending | A name for the business |
|---|---|---|---|---|---|
| 0 | 20 | 46 | 27 | 18 | Low income, careful spenders |
| 1 | 54 | 25 | 41 | 62 | Young, keen spenders |
| 2 | 40 | 33 | 86 | 82 | Affluent big spenders (target for premium offers) |
| 3 | 39 | 40 | 86 | 19 | Affluent but careful (persuade with value or quality) |
| 4 | 47 | 56 | 54 | 49 | Older, average spenders |

**Other clustering models:** DBSCAN with Week 3's `eps=0.7` found only one big group
plus 14 outliers. These segments are blobs of similar density, which suits K-Means
better than DBSCAN.

### Example C: breast cancer data (Week 2, built into scikit-learn)

- 569 rows, one per tumour, with 30 measurements. Answer: malignant / benign, so **classification**.
- Imbalanced (63% benign), so report **F1** and especially **recall for malignant**, because missing a cancer is the costly mistake.
- Ensembles do well here. Soft voting reached about 97% F1, but still missed 3 of 53 malignant tumours in the test set. That is the number to discuss in a report.

### Example D: Online Retail (Week 4, association rules)

- One row per **item** on an invoice, with the invoice number repeated, so this is **transactions**: association rules.
- Preparation is most of the work: remove cancelled invoices (starting with "C"), filter to one country, then pivot to one row per invoice with True/False per product.
- Note: the `Online_Retail.csv` in this folder is actually a copy of the README, not the data. Download the UCI "Online Retail" dataset to run Week 4.

### Example E: IMDB reviews (Week 6, text)

- One row per review, with a `sentiment` label, so this is **text classification**.
- The choices are in the preparation (cleaning, word counts vs TF-IDF, word pairs) as much as the model. Naive Bayes is the baseline; logistic regression on TF-IDF is usually best.

---

## 10. Picking two techniques (exam)

Many assessments ask for **two** techniques. Good pairings:

| Pairing | Why it works | Example with these files |
|---|---|---|
| **Two classifiers compared** | Simple and clear: an explainable model against a stronger one. | Practice data: decision tree vs random forest (or logistic regression) |
| **Clustering, then classification** | Find the segments, then predict which segment a new customer belongs to, or use the segment as an extra feature. | Mall data: K-Means segments, then a decision tree that explains the segment rules |
| **Two clustering methods compared** | See whether different methods find the same groups. | Mall data: K-Means vs Agglomerative, or vs DBSCAN |
| **NLP + classification** | Text needs both: cleaning and vectorising, then a classifier. | IMDB: Naive Bayes vs logistic regression |
| **Association rules + clustering** | Segment customers, then find what each segment buys together. | Online Retail: rules per country or per customer group |

Whatever you pick, justify it with the data: *"There is a labelled outcome column with
four balanced categories, so classification is appropriate..."*

---

## 11. One-page checklist

```
UNDERSTAND
[ ] Write down the business question in one sentence
[ ] Load the file and check: rows, columns, types, missing values, duplicates
[ ] What is one row? (person / item in a basket / document / time point)
[ ] Is there an answer column?   yes -> supervised   no -> unsupervised
[ ] Answer = categories -> classification;  number on a scale -> regression
[ ] No answer: groups -> clustering;  baskets -> association rules
[ ] Text? -> clean + vectorise first (Week 6)

PREPARE
[ ] Drop IDs, names, and anything that gives the answer away
[ ] Fix duplicates, missing values, impossible values, messy text
[ ] Encode categories (LabelEncoder for the target, get_dummies for features)
[ ] Split train/test BEFORE scaling (stratify for classification)
[ ] Scale for KNN, SVM, logistic regression, K-Means, DBSCAN

CHOOSE AND COMPARE
[ ] Know the baseline score (most common class / the average)
[ ] One simple, explainable model + one or two stronger ones
[ ] Same split and same score for all; cross-validation if possible
[ ] Pick the score that matches the cost of mistakes (accuracy / F1 / recall / R2 / silhouette / lift)
[ ] Check for overfitting (train score much higher than test score)

EXPLAIN
[ ] Best model and why (strength, weakness, what you did about it)
[ ] What drives the result (important columns, cluster profiles, top rules, top words)
[ ] A recommendation the business can act on
[ ] Limitations, and what you would try with more time
```
