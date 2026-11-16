# Exam Playbook: Using the Week Files as Your Toolkit

This is for an exam where you get a **new dataset and a business brief**, and you have
everything in this repository with you. The idea is simple: **you don't write a model
from scratch.** You work out what kind of problem it is, pick the week file that already
solves that kind of problem, copy it, and adapt it to the new data.

| File | Use it for |
|---|---|
| `Analysis_and_Model_Selection_Guide.md` | Deciding **what kind of problem** it is and **which models** suit it |
| `Starter_Template.py` / `.ipynb` | A quick **first look** at any CSV, and **cleaning** code to paste into a week file |
| `Week2.py`, `Week3.py`, `Week4.py`, `Week6.py` | **Working reference code** to copy and adapt |
| `README` | Plain-English model descriptions, cleaning snippets, report checklist |
| `Key_Functions_for_Adapting_Code.md` | What each function you need to change does, and what to change it to |

Everything in this playbook was tested by adapting each week file to a different
dataset. The errors quoted are the real ones you'll see.

---

## Contents

0. [Before the exam: prepare your toolkit](#0-before-the-exam-prepare-your-toolkit)
1. [The exam workflow at a glance](#1-the-exam-workflow-at-a-glance)
2. [Understand and classify the problem](#2-understand-and-classify-the-problem)
3. [Pick your reference file](#3-pick-your-reference-file)
4. [Copy and adapt: the general method](#4-copy-and-adapt-the-general-method)
5. [Adaptation cards: exactly what to change in each week file](#5-adaptation-cards-exactly-what-to-change-in-each-week-file)
6. [Cleaning: where it goes and what to add](#6-cleaning-where-it-goes-and-what-to-add)
7. [Using a model that isn't in the reference file](#7-using-a-model-that-isnt-in-the-reference-file)
8. [Two techniques: chaining week files together](#8-two-techniques-chaining-week-files-together)
9. [Error messages and their fixes](#9-error-messages-and-their-fixes)
10. [Worked example: a new brief from start to finish](#10-worked-example-a-new-brief-from-start-to-finish)
11. [Exam-day checklist](#11-exam-day-checklist)

---

## 0. Before the exam: prepare your toolkit

Do this **at home, days before**, not in the exam.

**1. Install everything and check it works**

```
python -m pip install numpy pandas matplotlib seaborn scikit-learn fuzzy-c-means mlxtend nltk beautifulsoup4 wordcloud
```

Then run each week file once. Week 6 also downloads word lists the first time it runs
(`nltk.download`, lines 18–20), which **needs the internet**. Run it before the exam in
case the exam room is offline.

**2. Get the datasets the week files expect**
- Week 4 needs the real UCI **Online Retail** dataset. The `Online_Retail.csv` in this repo is actually a copy of the README.
- Week 6 needs **`IMDB_Dataset.csv`**, which isn't in the repo.

**3. Fix the known problems in your copies now**

| File | Line(s) | Problem | Fix |
|---|---|---|---|
| `Week2.py` | 305–312 | The `results.append` / `print` lines aren't indented, so only the "Soft Voting" result is kept and the boxplot shows one box | Indent lines 305–312 by 4 spaces so they're inside the `for` loop |
| `Week4.py` | 16 | Reads `'Week 4/Online_Retail.csv'` (a sub-folder) | Change to the file's real location |
| `Week4.py` | 120–123 | `.map(hot_encode)` only works in pandas 2.1+. Older pandas gives `AttributeError: 'DataFrame' object has no attribute 'map'` | Use `.applymap(hot_encode)` on older pandas |
| `Week6.py` | 40 | Reads `'Week 6/IMDB_Dataset.csv'` | Change to the file's real location |
| `Week6.py` | 94 | `imdb.sample(2000, ...)` crashes if the data has fewer than 2,000 rows | `imdb.sample(min(2000, len(imdb)), random_state=42)` |

**4. Make an exam working folder.** Never edit the original week files during the exam.
Copy the one you need (e.g. `Week3.py` → `exam_task.py`) and keep the original open
side by side as a reference.

---

## 1. The exam workflow at a glance

```
 1  UNDERSTAND   Read the brief, write the business question in one sentence
 2  FIRST LOOK   Run Starter_Template Part 1 on the new file
 3  CLASSIFY     Guide sections 3-4: classification / clustering / rules / text?
 4  PICK         Choose the reference week file (section 3 of this playbook)
 5  ADAPT        Copy it, change the load line and column names (sections 4-5)
 6  CLEAN        Paste in the cleaning steps the data needs (section 6)
 7  MODEL        Run the existing models, or swap one in (section 7)
 8  EVALUATE     Choose the right score (guide section 7), screenshot outputs
 9  WRITE UP     Business question, data issues, methods, results, recommendation
```

A rough time split: **10%** understanding and first look, **25%** cleaning,
**35%** adapting and modelling, **30%** interpreting and writing up. The write-up earns
the marks, so don't spend all your time fixing code.

---

## 2. Understand and classify the problem

### 2.1 Read the brief for its key verbs

| The brief says... | Type of analysis | Reference |
|---|---|---|
| "predict whether / which / will they...", "identify customers likely to...", "classify" | Classification | `Week2.py` / Template 3A |
| "predict how much / how many / the price / the value" | Regression | Section 7 of this playbook |
| "segment", "group", "find types of customer", "profile" | Clustering | `Week3.py` |
| "bought together", "recommend products", "basket", "cross-sell", "bundle" | Association rules | `Week4.py` |
| "reviews", "comments", "tweets", "sentiment", "feedback text" | Text classification | `Week6.py` |
| "two techniques", "compare approaches" | Combine two | Section 8 |

### 2.2 Take a first look at the data

Open `Starter_Template.py` (or the notebook), set the five settings at the top, and run
it. Part 1 shows the shape, first rows, column types, summary statistics, missing values
and duplicates for **any** CSV. Screenshot it: these become the "data understanding"
part of your report.

### 2.3 Answer the guide's five questions

From section 3 of the guide. **Write the answers down**, because they become the
justification in your report:

1. **What is one row?** (a person, an item in a basket, a review...)
2. **Is there an answer column?** (yes = supervised, no = unsupervised)
3. **What does the answer column hold?** (categories = classification, a number on a scale = regression)
4. **What are the other columns like?** (IDs to drop, text categories to encode, impossible values, number columns stored as text)
5. **How big is it, and is it balanced?** (`df.shape`, `df["target"].value_counts()`)

Then follow the flowchart in section 4 of the guide.

---

## 3. Pick your reference file

| Your problem | Copy this | It already contains |
|---|---|---|
| **Classification**, simple and clean | `Starter_Template.py` Part 3A | Split X/y, encoding, train/test split, scaling, Decision Tree, Random Forest, Logistic Regression, KNN, report, confusion matrix, feature importance |
| **Classification** with ensembles and cross-validation | `Week2.py` | AdaBoost, Random Forest, Bagging (KNN), soft voting (SVMs), hard voting (KNNs), repeated cross-validation, boxplots |
| **Clustering** | `Week3.py` | Scaling, elbow and silhouette to choose k, K-Means, Fuzzy C-Means, DBSCAN, Agglomerative, cluster profiles |
| **Clustering**, K-Means only | `Starter_Template.py` Part 3B | Scaling, choosing k, K-Means, profiles, PCA plot |
| **Association rules** | `Week4.py` | Cleaning transactions, building baskets, 0/1 encoding, apriori, rules sorted by confidence and lift |
| **Text** | `Week6.py` | Text cleaning, word counts and TF-IDF, Naive Bayes, Logistic Regression, SGD, misclassified examples, word clouds, top words |
| **Regression** | (none) | Use the snippet in section 7 |

**Rule of thumb:** copy the file whose **data looks most like yours**, not the one with
the most models. A good plan for classification is to start with the template (fast and
reliable), then borrow Week 2's ensembles and cross-validation if you have time.

---

## 4. Copy and adapt: the general method

1. **Copy** the reference file to a new name, e.g. `exam_task.py`.
2. **Change the load line** to your file name (and `sep="\t"` or `sep=";"` if needed).
3. **Use the renaming trick.** Right after loading, rename your columns to the names the
   week file already uses. Then most of the file works **without further edits**. This
   is especially useful for Week 4 and Week 6, which use the same column names in many places:

   ```python
   data = data.rename(columns={"OrderID": "InvoiceNo", "Product": "Description",
                               "Qty": "Quantity", "Region": "Country"})
   ```

   (For Week 2 and Week 3, it's easier to change the few lines that build `X`, `y` or `features`; see section 5.)
4. **Paste in the cleaning** the data needs, right after loading (section 6).
5. **Run it, read the first error, fix it, run again.** Section 9 lists the errors you're likely to see.
6. **Delete what you don't need** (e.g. Week 4's extra countries, Week 2 activities you won't use) so your outputs stay focused.
7. **Rename variables only if it helps you explain the code.** `imdb` in Week 6 can stay `imdb`; it's just a name.

---

## 5. Adaptation cards: exactly what to change in each week file

Line numbers refer to the original files in this repository.

### 5.1 `Week2.py`: classification with ensembles

| Line(s) | What's there | Change it to |
|---|---|---|
| 41–55 | Loads the built-in breast cancer data into `X` and `y` | Your file (code below) |
| 57 | `print("Target classes:", encoder.classes_)` | Delete it, or `print(sorted(y.unique()))` if you didn't use an encoder |
| 66 | `test_size=0.25` | Keep, or `0.2` |
| 93–96 and 118–121 | `f1_score(y_test, predictions)` | Add `average="macro"` if there are **more than 2 classes** |
| 103–108 | Random Forest with `max_depth=1` | Remove `max_depth=1` for a real model (see note below) |
| 139–145 | `X_knn, y_knn = make_classification(...)` | `X_knn, y_knn = X, y` |
| 194–200 | `X_svm, y_svm = make_classification(...)` | `X_svm, y_svm = X, y` |
| 305–312 | Indentation bug | Indent by 4 spaces (section 0) |

Replacement for lines 41–55:

```python
df = pd.read_csv("new_data.csv")                 # add sep="\t" if needed
# ... cleaning from section 6 goes here ...
X = df.drop(columns=["Target"])                  # everything except the answer
X = pd.get_dummies(X, drop_first=True, dtype=int)  # text columns -> 0/1 columns
y = df["Target"]
if not pd.api.types.is_numeric_dtype(y):        # text answers -> numbers
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
```

**Things to know about Week 2:**
- **`X` must be all numbers with no gaps.** Any text column causes `could not convert string to float`; any gap causes `Input X contains NaN`.
- **Its models are set up for a teaching comparison, not for best results.** Activities 1–2 deliberately use one-question trees (`max_depth=1`) to compare boosting with bagging fairly. On the phone price data that gives AdaBoost **0.52** and Random Forest **0.61** accuracy, while a normal Random Forest gets about **0.88**. Say which version you used and why.
- **Week 2 doesn't scale the data.** KNN and SVM normally need scaling. Check both ways if you have time: on the phone data, unscaled KNN scored **0.92**, but the scaled KNN in the template scored **0.50**. RAM has by far the biggest numbers *and* is the most important column, so leaving it unscaled happened to help. Don't assume; compare and explain.
- **Activities 4 and 5 are slow-ish** (6 models × 30 cross-validation folds). On 2,000 rows they took a few seconds; on much larger data, reduce `n_repeats` in `RepeatedStratifiedKFold`.

*Tested on `Practice_Data.csv` (4 price bands): soft voting 0.960, the five SVMs
0.948–0.957, hard voting 0.930, bagged KNN 0.923.*

### 5.2 `Week3.py`: clustering

| Line(s) | What's there | Change it to |
|---|---|---|
| 32 | `pd.read_csv("Mall_Customers.csv")` | Your file |
| after 53 | (no cleaning, as the mall data was already clean) | Paste in cleaning (section 6). It is **needed** if any column has gaps or numbers stored as text. |
| 60 | `features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]` | Your **numeric** columns. No IDs, and no answer column if there is one. |
| 125 | `k = 5` | The k you chose from the elbow and silhouette table (lines 116–118 print it) |
| 247 | `DBSCAN(eps=0.7, min_samples=5)` | If you get 1 cluster or almost everything is noise (`-1`), try a smaller or bigger `eps` |

**Things to know about Week 3:**
- Line 25 changes to the script's own folder, so **put the data file next to the script**.
- **If `fuzzy-c-means` won't install,** comment out line 20 (`from fcmeans import FCM`) and the Fuzzy C-Means section, lines 149–231. The rest still runs.
- **Text features** (e.g. Gender) must be one-hot encoded (`pd.get_dummies`) or left out. Week 3 leaves Gender out.
- **Describe the clusters in the original units:** lines 131–132 already print the averages per cluster. That table is your business insight.

*Tested on a messy customer file: it failed with
`could not convert string to float: '£69.43'` until the cleaning block was added. After
that, the best silhouette was at k = 4 (0.292), so line 125 should be changed from 5 to 4.*

### 5.3 `Week4.py`: association rules

| Line(s) | What's there | Change it to |
|---|---|---|
| 16 | `pd.read_csv('Week 4/Online_Retail.csv')` | Your file, followed by the renaming trick (section 4) |
| 57 | Removes invoices containing `'C'` (cancellations) | Keep only if your data marks cancellations that way. If your IDs can contain the letter C for other reasons (e.g. `"INV-C12"`), delete this line. |
| 76–101 | Builds baskets for France, UK, Portugal and Sweden | Keep one block and change the country. Countries not in your data just give empty `(0, 0)` tables. |
| 120–123 | `.map(hot_encode)` | `.applymap(hot_encode)` on pandas older than 2.1 |
| 141 | `min_support=0.05` | Lower it (0.02, 0.01) if you get no rules |
| 148 | `metric="lift", min_threshold=1` | Usually keep |
| 156 | `print(rules_FR.head(10))` (columns get cut off) | `print(rules_FR[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))` |

**If your data has no country column**, build one basket table from all the data:

```python
basket = (data.groupby(['InvoiceNo', 'Description'])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('InvoiceNo'))
basket = basket.map(hot_encode)          # .applymap on older pandas
```

**If there's no quantity column** (each row just says an item was bought), add
`data['Quantity'] = 1` after loading.

**If the data is already one row per basket with 0/1 or True/False columns**, skip
straight to `apriori` (line 141).

*Tested on order data with different column names (`OrderID`, `Product`, `Qty`, `Region`):
one `rename` line was the only change needed. It also ran on old pandas once `.map` was
changed to `.applymap`.*

### 5.4 `Week6.py`: text classification

| Line(s) | What's there | Change it to |
|---|---|---|
| 40 | `pd.read_csv('Week 6/IMDB_Dataset.csv')` | Your file, then rename **and** relabel (code below) |
| 94 | `imdb.sample(2000, ...)` | `imdb.sample(min(2000, len(imdb)), random_state=42)` |
| 103 | Saves `imdb_clean_short_debug.csv` | Delete it if you don't want the extra file |
| 135 | Maps `'positive'` → 1, `'negative'` → 0 | No change if you relabel at load |
| 234–235 | Word clouds filter on `'positive'` / `'negative'` | No change if you relabel at load |

Rename the columns **and** turn your labels into `positive` / `negative`, and the rest of
the file works as it is:

```python
imdb = pd.read_csv('feedback.csv')
imdb = imdb.rename(columns={"text": "review", "label": "sentiment"})
imdb['sentiment'] = imdb['sentiment'].map({"pos": "positive", "neg": "negative"})
```

**Things to know about Week 6:**
- **Cleaning removes "not" and "no"** (they're stopwords), so "not good" loses its meaning. You can mention this as a limitation, or keep them with `stop_words = set(stopwords.words('english')) - {"not", "no", "nor"}` at line 32.
- **The vectorisers are fitted on all the data before the split** (lines 123–124), which slightly leaks test information. Acknowledge it in your limitations.
- **More than 2 classes** (e.g. positive / neutral / negative): change line 135 to map to 0/1/2, and note that `show_top_coef` (line 258) and the two word clouds assume two classes.

*Tested on an 800-row feedback file with `text`/`label` columns and `pos`/`neg` labels:
it crashed at line 94 until the sample size fix was made, then ran end to end.*

---

## 6. Cleaning: where it goes and what to add

**Where:** straight after the data is loaded, before any `X`, `y` or `features` line.

**What:** copy only the steps your data needs from `Starter_Template.py` Part 2 (or
README section 4). Each step below is matched to what you saw in the first look:

| You saw... | Add this step |
|---|---|
| `df.duplicated().sum()` > 0 | `df = df.drop_duplicates()` |
| ID or name columns | `df = df.drop(columns=["CustomerID"])` |
| A number column showing as `object` (e.g. `"£69.43"`) | `df["Spend"] = pd.to_numeric(df["Spend"].str.replace("£", ""), errors="coerce")` |
| Messy categories (`" Male"`, `"male"`) | `df["Gender"] = df["Gender"].str.strip().str.lower()` |
| Impossible zeros (screen width 0) | `df["sc_w"] = df["sc_w"].replace(0, np.nan)`, then fill |
| Missing numbers | `df["Age"] = df["Age"].fillna(df["Age"].median())` |
| Missing text | `df["City"] = df["City"].fillna(df["City"].mode()[0])` |
| Extreme values (age 999) | The IQR filter (Template Step 7) |
| Missing answers (target) | `df = df.dropna(subset=["Target"])` |

**Keep a cleaning log as you go** (what you found, what you did, why, and how many rows
it affected). The report's "data issues" section is then already written. Example:
*"5 duplicate rows removed; 30 missing ages filled with the median because age had an
extreme outlier (999), which was then removed with the IQR rule."*

---

## 7. Using a model that isn't in the reference file

Every scikit-learn model works the same way, so you can drop a different one into any
week file:

```python
from sklearn.ensemble import RandomForestClassifier     # 1. import
model = RandomForestClassifier(random_state=42)         # 2. create
model.fit(X_train, y_train)                             # 3. learn
y_pred = model.predict(X_test)                          # 4. predict
print(accuracy_score(y_test, y_pred))                   # 5. score
```

**Course models and their import lines:**

| Model | Import | Scale first? |
|---|---|---|
| Logistic Regression | `from sklearn.linear_model import LogisticRegression` | Yes |
| Decision Tree | `from sklearn.tree import DecisionTreeClassifier` | No |
| Random Forest | `from sklearn.ensemble import RandomForestClassifier` | No |
| KNN | `from sklearn.neighbors import KNeighborsClassifier` | Yes |
| SVM | `from sklearn.svm import SVC` | Yes |
| AdaBoost / Bagging / Voting | `from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, VotingClassifier` | Depends |
| Naive Bayes (text) | `from sklearn.naive_bayes import MultinomialNB` | No |
| SGD (text) | `from sklearn.linear_model import SGDClassifier` | Yes |
| K-Means / DBSCAN / Agglomerative | `from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering` | Yes |
| Linear Regression | `from sklearn.linear_model import LinearRegression` | Recommended |
| Random Forest Regressor | `from sklearn.ensemble import RandomForestRegressor` | No |

**Regression (predicting a number), the one type with no week file.** Start from the
template's classification part and swap in:

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

X = df.drop(columns=["Price"])
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)        # no stratify for numbers

for name, model in {"Linear Regression": LinearRegression(),
                    "Random Forest Regressor": RandomForestRegressor(random_state=42)}.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"{name:24s} R2 = {r2_score(y_test, pred):.3f}  MAE = {mean_absolute_error(y_test, pred):.1f}")
```

R² near 1 is good. R² near 0 (or below) means the inputs can't predict the number; that
is a finding to report, not a bug.

---

## 8. Two techniques: chaining week files together

A strong combination is **clustering, then classification**: find the segments with Week 3,
then train a Week 2-style decision tree to predict each customer's segment. The tree
shows which columns define the segments, and can assign new customers to a segment.

```python
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Part 1: Week 3, find the segments
features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
X_scaled = StandardScaler().fit_transform(df[features])
df["cluster"] = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(X_scaled)

# Part 2: Week 2, learn to predict the segment
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df["cluster"], test_size=0.25, random_state=1, stratify=df["cluster"])
tree = DecisionTreeClassifier(max_depth=3, random_state=1).fit(X_train, y_train)
print("Accuracy:", accuracy_score(y_test, tree.predict(X_test)))
print(pd.Series(tree.feature_importances_, index=features))
```

*Tested on the mall data: the tree predicts the segment with 0.98 accuracy, and spending
score is the most important column (0.45).*

Other pairings (see section 10 of the guide): two classifiers compared on the same
split; K-Means compared with DBSCAN or Agglomerative; association rules for one segment
of customers at a time.

---

## 9. Error messages and their fixes

Read the **last line** of the error first. ✓ = seen while testing the week files on new data.

| Error (last line) | Cause | Fix |
|---|---|---|
| `FileNotFoundError: ... No such file` | Wrong name or folder | Check spelling and `.csv`; put the file next to the script; or use the full path `r"C:\...\file.csv"` |
| Everything ends up in one column | Wrong separator | `sep="\t"` or `sep=";"` |
| `KeyError: 'Age'` | Column name spelt differently (spaces, capitals) | `print(df.columns.tolist())` and copy the exact name |
| ✓ `ValueError: could not convert string to float: '£69.43'` | A text value in a column the model needs as a number | `pd.to_numeric(...)` for numbers stored as text; `pd.get_dummies` for categories |
| `ValueError: Input X contains NaN` | Missing values reached the model | Fill or drop them (section 6) |
| ✓ `ValueError: Target is multiclass but average='binary'` | `f1_score` with more than 2 classes | `f1_score(y_test, y_pred, average="macro")` |
| ✓ `ValueError: Cannot take a larger sample than population` | `sample(2000)` on a smaller dataset (Week 6) | `sample(min(2000, len(imdb)), ...)` |
| ✓ `AttributeError: 'DataFrame' object has no attribute 'map'` | Older pandas (Week 4) | `.applymap(hot_encode)` |
| ✓ `ValueError: The least populated classes in y have only 1 member` | `stratify=y` with a class that has only 1 row | Remove that class, merge rare classes, or drop `stratify` |
| ✓ `ValueError: The input DataFrame df containing the frequent itemsets is empty` | `min_support` too high, so apriori found nothing | Lower `min_support` (0.02, 0.01) |
| `ModuleNotFoundError: No module named 'fcmeans'` (or `mlxtend`, `bs4`, ...) | Package not installed | `python -m pip install fuzzy-c-means` (install name ≠ import name, see README) |
| `LookupError: Resource stopwords not found` | NLTK word lists not downloaded | `nltk.download('stopwords')` (needs internet; do it before the exam) |
| `ConvergenceWarning` | Logistic regression needs more steps | `LogisticRegression(max_iter=1000)` and scale the data |
| Output shows `...` instead of columns | pandas hides wide tables | `pd.set_option('display.max_columns', None)` at the top |
| Charts don't appear | Script ended or chart window hidden | Make sure `plt.show()` is there; close each chart window to continue |

---

## 10. Worked example: a new brief from start to finish

> *Brief:* "A phone retailer wants to predict which **price band** (0–3) a new phone
> belongs to from its specification, and to know **which features drive price**."
> *Data:* `Practice_Data.csv`

| Step | What you do | Result |
|---|---|---|
| 1 Understand | Business question: "Predict a phone's price band and find what drives it." The key verb "predict" plus a category means classification. | |
| 2 First look | Template Part 1 with `SEPARATOR = "\t"`, `TARGET_COLUMN = "price_range"` | 2,000 rows, 21 columns, no gaps, no duplicates, 4 balanced classes (500 each) |
| 3 Classify | Guide Q1–Q5: one row = one phone; the answer column holds 4 categories; balanced | **Classification**; accuracy is fine because the classes are balanced; baseline 25% |
| 4 Pick | Template 3A for the main comparison; Week 2 for ensembles | |
| 5 Adapt | Week 2: replace lines 41–55, add `average="macro"` (4 classes), point Activities 3–5 at `X, y`, fix the indentation bug | Runs |
| 6 Clean | `sc_w` (screen width) is 0 for 180 phones, which is impossible, so treat it as missing and fill with the median | Cleaning log entry |
| 7 Model | Template (with the `sc_w` fix): Logistic Regression **0.963**, Random Forest 0.895, Decision Tree 0.830, KNN 0.527. Week 2: soft voting **0.960** (cross-validated) | |
| 8 Evaluate | Classification report and confusion matrix for the best model; random forest feature importance | RAM is by far the strongest driver, then battery power and screen pixels |
| 9 Write up | "Logistic regression was chosen: as accurate as the ensembles, faster and explainable. RAM drives price band most, so..." | Report |

---

## 11. Exam-day checklist

```
BEFORE YOU START
[ ] Packages work; week files run; NLTK data downloaded; datasets in place
[ ] Known bugs fixed in your copies (section 0)
[ ] A fresh working folder; originals kept open as reference

UNDERSTAND (first ~10%)
[ ] Business question written in one sentence
[ ] Template Part 1 run and screenshotted
[ ] Guide Q1-Q5 answered and written down -> type of analysis decided

BUILD
[ ] Reference file chosen and copied (never edit the original)
[ ] Load line changed; columns renamed to match, or X/y/features lines changed
[ ] Cleaning pasted in after loading; cleaning log kept
[ ] Week-specific changes made (adaptation card, section 5)
[ ] Runs end to end; each error fixed using section 9

EVALUATE
[ ] Right score for the problem (accuracy / F1 / recall / silhouette / lift)
[ ] Compared with the baseline; checked for overfitting (train vs test)
[ ] Key outputs and charts screenshotted

WRITE UP
[ ] Business problem -> data issues and fixes (WHY) -> techniques and why
    -> results -> business recommendation -> limitations
```
