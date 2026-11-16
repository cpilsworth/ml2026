# Key Functions to Change When Adapting the Week Files

When you reuse a week file on a new dataset, **most of the code stays exactly as it is.**
The model maths, the `fit` and `predict` calls and the scoring functions don't care what
your data is about. What *does* need changing is a small set of function calls that
mention **your data**: the file name, the column names, the labels, and a handful of
settings that depend on how big and what shape your data is.

This guide explains each of those key functions in plain English: what it does, the
line it's on in the original week file, what to change it to, and what goes wrong if you
don't. It goes with:
- `Exam_Playbook.md`, the step-by-step process;
- `Analysis_and_Model_Selection_Guide.md`, for deciding which week file fits your problem.

Line numbers refer to the original files in this repository.

---

## Contents

1. [Reading a line of Python (2-minute version)](#1-reading-a-line-of-python-2-minute-version)
2. [The three kinds of change](#2-the-three-kinds-of-change)
3. [Type A: where your data is and what things are called](#3-type-a-where-your-data-is-and-what-things-are-called)
4. [Type B: getting the data into the form the model needs](#4-type-b-getting-the-data-into-the-form-the-model-needs)
5. [Type C: settings that depend on your data](#5-type-c-settings-that-depend-on-your-data)
6. [Results and charts that name the old data](#6-results-and-charts-that-name-the-old-data)
7. [Functions you can leave alone](#7-functions-you-can-leave-alone)
8. [Quick lookup: "my new data has..."](#8-quick-lookup-my-new-data-has)

---

## 1. Reading a line of Python (2-minute version)

```python
df = pd.read_csv("Mall_Customers.csv", sep=",")
```

| Part | Meaning |
|---|---|
| `df =` | "Store the result under the name `df`." `df` is short for "data frame", a table. |
| `pd.read_csv` | The `read_csv` function from the **pandas** library (nicknamed `pd` by `import pandas as pd`) |
| `( ... )` | The information you hand to the function (its **arguments**) |
| `"Mall_Customers.csv"` | Text (a "string"). **This is the kind of thing you change.** |
| `sep=","` | A **named setting**. Optional settings have a default if you leave them out. |

Four other things you'll see:

- **`df["Age"]`** picks the column called Age. `df[["Age", "Income"]]` (double brackets) picks several.
- **`["Age", "Income"]`** is a **list** (items in order); **`{"old": "new"}`** is a **dictionary** (look up a value by its name).
- **`df.drop_duplicates()`** is a **method**: an action belonging to `df`, written after a dot.
- **Most table actions give back a *new* table** and leave the original alone. That's why you see `df = df.drop_duplicates()`. Leave out the `df =` and nothing changes.

---

## 2. The three kinds of change

| Kind | What changes | Key functions | Section |
|---|---|---|---|
| **A. Where and what** | File name; column names; which column is the answer; what the labels are called | `pd.read_csv`, `df.rename`, `df[...]`, `df.drop`, `Series.map`, `LabelEncoder` | 3 |
| **B. Form** | Converting your data into what the model accepts: numbers with no gaps, a 0/1 basket table, or clean text | `fillna`, `pd.to_numeric`, `pd.get_dummies`, `StandardScaler`, `groupby/unstack`, `apply` | 4 |
| **C. Settings** | Numbers inside model functions that depend on your data's size and shape | `n_clusters`, `eps`, `min_support`, `average`, `test_size`, `max_depth`, `max_features` | 5 |

**A useful way to spot what to change:** scan the week file for anything in quotation
marks (`"Mall_Customers.csv"`, `"France"`, `'positive'`) and any number written into a
function call (`k = 5`, `eps=0.7`, `min_support=0.05`). Almost everything you need to
change is one or the other.

---

## 3. Type A: where your data is and what things are called

### 3.1 `pd.read_csv()`: load your file

**What it does:** reads a CSV file into a table (a DataFrame).

**How to change it:** put in your file name. Add `sep="\t"` if the file uses tabs, or
`sep=";"` for semicolons. You know you need `sep` if `df.head()` shows everything
squashed into **one** column. Add `encoding="latin1"` if you get a `UnicodeDecodeError`.

| File | Line | Original | Change to |
|---|---|---|---|
| `Week3.py` | 32 | `pd.read_csv("Mall_Customers.csv")` | `pd.read_csv("your_file.csv")` |
| `Week4.py` | 16 | `pd.read_csv('Week 4/Online_Retail.csv')` | `pd.read_csv('your_file.csv')` (the original points to a sub-folder that doesn't exist here) |
| `Week6.py` | 40 | `pd.read_csv('Week 6/IMDB_Dataset.csv')` | `pd.read_csv('your_file.csv')` |
| `Week2.py` | 41 | `load_breast_cancer()` | See 3.2 |

**Where the file must be:** `Week3.py` line 25 (`os.chdir(...)`) makes the script look in
**its own folder**, so put the CSV next to the script. The other files look in whatever
folder you run them from; the simplest fix is also to keep the CSV next to the script.

### 3.2 `load_breast_cancer()` and `make_classification()`: built-in and made-up data (Week 2)

**What they do:** `load_breast_cancer()` loads a dataset that comes with scikit-learn.
`make_classification()` **invents** a random practice dataset. Neither reads a file, so
for new data you replace them with `pd.read_csv` and your own `X` and `y`.

| Lines | Original | Change to |
|---|---|---|
| 41–55 | Loads breast cancer data into `X` and `y` | `df = pd.read_csv(...)`, then build `X` and `y` (3.4, 3.5) |
| 139–145 | `X_knn, y_knn = make_classification(...)` | `X_knn, y_knn = X, y` |
| 194–200 | `X_svm, y_svm = make_classification(...)` | `X_svm, y_svm = X, y` |

### 3.3 `df.rename(columns={...})`: the renaming trick

**What it does:** gives columns new names. The dictionary reads `{"current name": "new name"}`.

**How it solves the problem:** Week 4 and Week 6 mention their column names (like
`'InvoiceNo'` or `'review'`) in many places. Instead of finding and changing every one,
**rename your columns to the names the week file already uses**, straight after
loading. Then the rest of the file works unchanged.

```python
# Week 4: your columns are OrderID, Product, Qty, Region
data = data.rename(columns={"OrderID": "InvoiceNo", "Product": "Description",
                            "Qty": "Quantity", "Region": "Country"})

# Week 6: your columns are text and label
imdb = imdb.rename(columns={"text": "review", "label": "sentiment"})
```

*Tested: for Week 4, this one line was the only change needed to run on differently
named order data.*

**Names the week files expect:**

| File | Column names it uses |
|---|---|
| `Week4.py` | `InvoiceNo` (basket ID), `Description` (item), `Quantity`, `Country` |
| `Week6.py` | `review` (the text), `sentiment` (the label) |
| `Week3.py` | Whatever is in the `features` list (line 60), so change the list instead (3.4) |
| `Week2.py` | None; you build `X` and `y` yourself (3.4) |

### 3.4 Choosing columns: `df[features]`, `df.drop(columns=[...])`

**What they do:** `df[list_of_names]` **keeps** only those columns;
`df.drop(columns=list_of_names)` **removes** them.

**For clustering (Week 3, lines 60–61):** list the number columns to group by.

```python
features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]   # original
features = ["Age", "Spend", "Tenure"]                                 # your columns
X = df[features].copy()
```

Leave out ID numbers and any "answer" column. Use only **number** columns (or convert
text columns first, see 4.2).

**For classification (Week 2, the template):** split the table into the inputs (`X`)
and the answer (`y`).

```python
X = df.drop(columns=["Churn"])     # everything except the answer
y = df["Churn"]                    # just the answer
```

Spell column names **exactly** as they appear: capitals, spaces and brackets all count.
A spelling mistake gives `KeyError: 'Churn'`. Run `print(df.columns.tolist())` to copy
the exact names.

### 3.5 Turning answer labels into numbers: `Series.map({...})` and `LabelEncoder`

Models need the answer as numbers. There are two ways to convert it:

**`df["col"].map({...})`** swaps each value using a dictionary you write. You choose
exactly which label gets which number.

```python
# Week 6, line 135 (original)
y = imdb['sentiment'].map({'positive': 1, 'negative': 0}).values
```

If your labels are different (`"pos"/"neg"`, `"Good"/"Bad"`), either change this
dictionary **or**, better, convert your labels to `'positive'`/`'negative'` straight
after loading:

```python
imdb['sentiment'] = imdb['sentiment'].map({"pos": "positive", "neg": "negative"})
```

Week 6 also uses the words `'positive'` and `'negative'` in its word clouds
(lines 234–235) and when printing mistakes (line 223), so relabelling once at the start
fixes all of them. ⚠️ Any label missing from the dictionary becomes a gap (`NaN`), so
include every label.

**`LabelEncoder().fit_transform(y)`** (Week 2, lines 54–55) numbers the labels
**automatically, in alphabetical order** (`"No"`=0, `"Yes"`=1). It's quicker, but you
don't choose the order. Check `encoder.classes_` to see which number means what.

If the answer column **already holds numbers** (like `price_range` 0–3), you don't need
either. Line 57 of Week 2 prints `encoder.classes_`, so delete it or it will fail.

### 3.6 Choosing rows: `df[condition]` and `.str.contains()`

**What they do:** keep only the rows where a condition is true.

| File | Line | Original | What to check for your data |
|---|---|---|---|
| `Week4.py` | 76 | `data[data['Country'] == "France"]` | Change `"France"` to a value that exists in **your** data (`print(data['Country'].unique())` lists them). If you have no country column, see 4.4. |
| `Week4.py` | 57 | `data[~data['InvoiceNo'].str.contains('C')]` | Removes invoices containing the letter "C" (cancellations in the original data). `~` means "NOT". If your IDs can contain a C for other reasons, **delete this line**. |

---

## 4. Type B: getting the data into the form the model needs

Every model needs a particular form of data:

| Week file | Needs |
|---|---|
| `Week2.py`, `Week3.py` | A table of **numbers only**, with **no gaps** |
| `Week4.py` | One row per **basket**, one **0/1 column per item** |
| `Week6.py` | A column of **text** and a column of **labels** |

The original datasets were already in that form, so the week files have little or no
cleaning. **New data usually isn't**, and the error message tells you what's wrong:

| Error you see | What it means | Function that fixes it (below) |
|---|---|---|
| `could not convert string to float: '£69.43'` | Text in a column the model needs as numbers | 4.1 `pd.to_numeric` or 4.2 `pd.get_dummies` |
| `Input X contains NaN` | Gaps (missing values) | 4.1 `fillna` / `dropna` |
| `could not convert string to float: 'Leeds'` | A category column (a word, not a number in disguise) | 4.2 `pd.get_dummies` |

**Where to put the cleaning:** straight after loading, before the lines that build `X`,
`y` or `features`. In `Week3.py` that means after line 53.

### 4.1 Cleaning functions

| Function | Plain meaning | When your new data needs it |
|---|---|---|
| `df = df.drop_duplicates()` | Remove rows that are exact copies | `df.duplicated().sum()` is more than 0 |
| `df = df.drop(columns=["ID"])` | Remove columns | IDs, names, free-text notes |
| `df = df.dropna(subset=["Target"])` | Remove rows with a gap in that column | Rows with no answer, or only a few gaps overall |
| `df["Age"] = df["Age"].fillna(df["Age"].median())` | Fill gaps in a number column with the middle value | Gaps in numbers (the median isn't thrown off by extreme values) |
| `df["City"] = df["City"].fillna(df["City"].mode()[0])` | Fill gaps in a text column with the most common value | Gaps in categories |
| `pd.to_numeric(df["Spend"].str.replace("£", ""), errors="coerce")` | Remove the £, then turn the text into numbers. Anything unconvertible becomes a gap. | A number column that `df.info()` shows as text |
| `df["Gender"].str.strip().str.lower()` | Remove spaces and make lowercase, so " Male" and "male" match | Messy categories |
| `df["sc_w"].replace(0, np.nan)` | Turn impossible zeros into gaps (then fill them) | `describe()` shows 0 where 0 can't happen |
| `quantile(0.25)`, `quantile(0.75)` and the IQR filter | Remove extreme values | `describe()` shows impossible maximums (age 999) |

Ready-to-paste versions are in `Starter_Template.py`, Steps 3–7.

### 4.2 `pd.get_dummies()`: categories into 0/1 columns

**What it does:** turns each text category into its own 0/1 column ("one-hot
encoding"): `City` becomes `City_Leeds` and `City_York`.

**How it solves the problem:** models can't use words like "Leeds", but they can use
"Leeds: 1, York: 0".

```python
X = pd.get_dummies(X, drop_first=True, dtype=int)
```

- `drop_first=True` drops one column per category, because it's implied by the others.
- `dtype=int` gives 0/1 instead of True/False.

Use it on the **inputs** (`X`). For the **answer** (`y`), use 3.5 instead. Week 3
avoids this by leaving Gender out of its `features`; you can do the same.

### 4.3 `StandardScaler()`: put columns on the same scale

**What it does:** rescales every number column to average 0 and spread 1.

**Why it matters for new data:** methods that measure **distance** (KNN, SVM, K-Means,
DBSCAN) are dominated by whichever column has the biggest numbers. Week 3 already scales
(lines 68–69). **Week 2 doesn't scale at all**, so with new data, consider adding it:

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # learn the scaling from the training data
X_test = scaler.transform(X_test)         # apply the same scaling to the test data
```

⚠️ `fit_transform` on training data only, then `transform` on test data. Fitting on all
the data lets test information leak in.

⚠️ Scaling isn't always better. On the phone price data, unscaled KNN scored 0.92 but
scaled KNN 0.50: RAM had the biggest numbers *and* was the most useful column. Try both
and explain what you found.

### 4.4 Building baskets (Week 4): `groupby`, `sum`, `unstack`, `fillna`, `set_index`

Lines 76–79 turn a long list of "invoice, item, quantity" rows into a **basket table**:

```python
basket_France = (data[data['Country'] == "France"]     # 1. keep one country
                 .groupby(['InvoiceNo', 'Description']) # 2. group by invoice AND item
                 ['Quantity'].sum()                    # 3. total quantity of each item per invoice
                 .unstack()                            # 4. items become COLUMNS: one row per invoice
                 .reset_index().fillna(0)              # 5. blanks (item not bought) become 0
                 .set_index('InvoiceNo'))              # 6. invoice numbers as row labels
```

Then `hot_encode` (lines 111–116) with `.map(hot_encode)` (lines 120–123) turns
quantities into 0 or 1, because apriori only cares whether an item was bought, not how many.

**Adapting it:**

| Your data | Change |
|---|---|
| Different column names | Rename them (3.3); nothing here changes |
| A different country | Change `"France"` on line 76 |
| No country column | Delete the first step: `basket = (data.groupby(['InvoiceNo', 'Description'])...` |
| No quantity column | Add `data['Quantity'] = 1` after loading |
| Already one row per basket with 0/1 columns | Skip all of this and go straight to `apriori` (line 141) |
| Older pandas (before 2.1) | `.map(hot_encode)` fails with `'DataFrame' object has no attribute 'map'`; use `.applymap(hot_encode)` |

Lines 83–101 repeat the same steps for the UK, Portugal and Sweden. Delete them or
change the countries; a country that isn't in your data just gives an empty `(0, 0)` table.

### 4.5 Preparing text (Week 6): `.apply(clean_text)` and friends

**`imdb['review'].apply(clean_text)`** (line 100) runs the `clean_text` function on
every row of the text column. With the renaming trick (3.3) it works on your text
unchanged. Inside `clean_text`:

| Line | Function | Plain meaning | Consider changing when... |
|---|---|---|---|
| 67 | `BeautifulSoup(text, "html.parser").get_text()` | Strips HTML tags like `<br />` | Never harmful; leave it |
| 74 | `re.sub(r'[^a-zA-Z\s]', ' ', text)` | Removes everything that isn't a letter or space | Numbers matter in your text (ratings, sizes). Change line 100 to `imdb['review'].apply(lambda t: clean_text(t, remove_digits=False))` to keep them. |
| 32 | `stopwords.words('english')` | The list of filler words to remove | Negation matters: the list includes "not" and "no". *Tested:* "Not good, only 2 stars" is cleaned to **"good star"**, the opposite meaning. Change line 32 to `stop_words = set(stopwords.words('english')) - {"not", "no", "nor"}` to get "not good star". |
| 86 | `lemmatizer.lemmatize(t)` | Reduces a word to its dictionary form | It treats every word as a noun: "movies" → "movie", but "running" stays "running" (`lemmatize("running", pos="v")` gives "run") |

**`imdb.sample(2000, random_state=42)`** (line 94) picks 2,000 random rows to test the
cleaning on. ⚠️ It **crashes** if your data has fewer than 2,000 rows. Change it to
`imdb.sample(min(2000, len(imdb)), random_state=42)`.

---

## 5. Type C: settings that depend on your data

These are the numbers written inside function calls. The original values were chosen
for the original data and may be wrong for yours.

### 5.1 Splitting and scoring

| Function (file, line) | Setting | What it controls | How to choose for new data |
|---|---|---|---|
| `train_test_split` (W2 66, W6 139) | `test_size=0.25` / `0.2` | Share of rows held back for testing | 0.2–0.25 is normal; keep it |
| | `stratify=y` | Keeps the class mix the same in both parts | Keep it for classification. ⚠️ Fails if a class has only 1 row: remove that class or drop `stratify`. Leave it out when predicting a number. |
| `f1_score` (W2 93–96, 118–121) | `average` (not set, so two classes are assumed) | How F1 handles the classes | **More than 2 classes: add `average="macro"`**, or you get `Target is multiclass but average='binary'` |
| `cross_val_score` (W2 171) | `scoring="accuracy"` | Which score to report | Imbalanced classes: `"f1"` (2 classes) or `"f1_macro"` (more) |
| `RepeatedStratifiedKFold` (W2 165) | `n_splits=10, n_repeats=3` | 30 test runs per model | Large data or short on time: `n_repeats=1` (10 runs) |

### 5.2 Classification models

| Function (file, line) | Setting | What it controls | How to choose for new data |
|---|---|---|---|
| `RandomForestClassifier` (W2 103) | `max_depth=1` | How deep each tree may go | **Remove it for a real model.** Week 2 uses one-question trees on purpose, to compare with AdaBoost. On the phone data that scores 0.61; a normal forest about 0.88. |
| `AdaBoostClassifier` (W2 78) | `estimator=DecisionTreeClassifier(max_depth=1)`, `n_estimators=200` | The small model it builds up, and how many | Usually keep; say in the report that it uses stumps |
| `KNeighborsClassifier` | `n_neighbors` (default 5) | How many neighbours vote | Week 2 compares 1, 3, 5, 7, 9; copy that loop to choose |
| `SVC` (W2 208+) | `kernel="poly"`, `degree=1…5` | The shape of the boundary between classes | Keep Week 2's comparison of degrees |
| `LogisticRegression` (W6 174) | `max_iter=400` | How many steps it may take to settle | Raise it (e.g. 1,000) if you see a `ConvergenceWarning` |
| `DecisionTreeClassifier` | `max_depth` | How many questions deep | 3–5 for an explainable tree; unlimited memorises the training data |

### 5.3 Clustering

| Function (file, line) | Setting | What it controls | How to choose for new data |
|---|---|---|---|
| `range(2, 11)` (W3 81) | k values to try | Which group counts the elbow and silhouette charts test | Fine as it is |
| `k = 5` (W3 125) | Number of groups | Used by K-Means, Fuzzy C-Means and Agglomerative | **Change it** to the best k from the table printed at line 118. *Tested: on a new customer file the best was 4, not 5.* |
| `KMeans` (W3 126) | `n_clusters=k, n_init=10` | k, and how many starting positions to try | Only k changes |
| `FCM` (W3 156) | `n_clusters=k` | Fuzzy version of the same | Follows `k` |
| `[3, 4, 5, 6]` (W3 186) | Fuzzy k values compared | | Include your chosen k |
| `DBSCAN` (W3 247) | `eps=0.7, min_samples=5` | How close points must be to count as a crowd | **Depends on your data.** One big cluster: lower `eps`. Almost all noise (`-1`): raise `eps`. |
| `AgglomerativeClustering` (W3 266) | `n_clusters=k` | Follows `k` | |

### 5.4 Association rules

| Function (file, line) | Setting | What it controls | How to choose for new data |
|---|---|---|---|
| `apriori` (W4 141) | `min_support=0.05` | Only combinations in at least 5% of baskets | **No rules, or the error "frequent itemsets is empty"?** Lower it to 0.02 or 0.01. Thousands of rules? Raise it. |
| `association_rules` (W4 148) | `metric="lift", min_threshold=1` | Keeps rules with lift of at least 1 | Usually keep |
| `sort_values` (W4 152) | `['confidence', 'lift']` | Order of the printed rules | Keep |

### 5.5 Text

| Function (file, line) | Setting | What it controls | How to choose for new data |
|---|---|---|---|
| `MAX_FEAT = 5000` (W6 110) | Vocabulary size | How many of the most common words (and pairs) become columns | Small data (a few hundred texts): 1,000–2,000 is plenty |
| `CountVectorizer` / `TfidfVectorizer` (W6 115, 120) | `ngram_range=(1, 2)` | Single words plus word pairs | Keep; pairs capture "not good" (if "not" survives cleaning, see 4.5) |
| `y = ...map({...})` (W6 135) | Label numbers | | **More than 2 labels:** `{'negative': 0, 'neutral': 1, 'positive': 2}`. Then add `average="macro"` anywhere F1 is used. The word clouds and `show_top_coef` (line 258) assume two classes. |

---

## 6. Results and charts that name the old data

These lines produce your results. Some update themselves; some contain the old data's
names typed in as fixed text.

| File, line | Function | Updates itself? | What to do |
|---|---|---|---|
| W3 132, 229 | `df.groupby("kmeans_cluster")[features].mean()` | ✅ Uses your `features` list | Nothing. This is your cluster profile table. |
| W3 140–141, 211–212 | `sns.scatterplot(x=features[1], y=features[2], ...)` | ✅ Uses the 2nd and 3rd features | ⚠️ Needs **at least 3** features, or you get `IndexError: list index out of range`. Use `features[0]` and `features[1]` if you have only 2. |
| W3 142–144, 213–215 | `plt.title("... (K=5)")`, `plt.xlabel("Annual Income (k$)")`, `plt.ylabel("Spending Score (1-100)")` | ❌ Typed in | Change the text to your column names and k. Otherwise the charts are labelled wrongly. |
| W4 155–156 | `print("... for France")`, `rules_FR.head(10)` | Partly | Change "France" in the message. To see every column: `print(rules_FR[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))` |
| W6 234–235, 244, 252 | Word clouds and their titles | ✅ If you relabelled to `'positive'`/`'negative'` (3.5) | Otherwise change the labels in lines 234–235 |
| W6 258–278 | `show_top_coef(lr, bow_vec)` | ✅ Reads your vocabulary | Two classes only |
| W2 322–323, 408–409 | `plt.title("Soft Voting vs Individual SVM Models")` | ✅ Still accurate | Nothing |
| Template Step 14 | `pd.Series(model.feature_importances_, index=X.columns)` | ✅ Uses your column names | Nothing |

---

## 7. Functions you can leave alone

These do the same job whatever the data, so you normally **don't** touch them:

| Function | What it does |
|---|---|
| `model.fit(X_train, y_train)` | The model learns from the training data |
| `model.predict(X_test)` | The model gives answers for new rows |
| `kmeans.fit_predict(X_scaled)` | Learn the clusters and label every row in one step |
| `scaler.fit_transform(X)` / `scaler.transform(X)` | Scale data (just make sure it's fitted on the right data) |
| `accuracy_score`, `classification_report`, `confusion_matrix`, `silhouette_score` | Scores and reports |
| `cross_val_score(...)` | Repeated testing (only `scoring` may change) |
| `apriori(...)`, `association_rules(...)` | Only their settings change (5.4) |
| `print(...)`, `plt.show()`, `plt.tight_layout()`, `df.head()`, `df.info()`, `df.describe()` | Showing results |

**Adding a model that isn't in the week file** works the same way: import it, create
it, then call `fit` and `predict`. Section 7 of `Exam_Playbook.md` has the import lines.

---

## 8. Quick lookup: "my new data has..."

| My new data... | Change this function | Section |
|---|---|---|
| ...is a different file | `pd.read_csv("new.csv")` | 3.1 |
| ...is tab or semicolon separated | `pd.read_csv(..., sep="\t")` / `sep=";"` | 3.1 |
| ...is replacing Week 2's built-in data | Replace `load_breast_cancer()` / `make_classification()` | 3.2 |
| ...has different column names (Week 4/6) | `df.rename(columns={...})` after loading | 3.3 |
| ...has different columns to cluster on | `features = [...]` (Week 3, line 60) | 3.4 |
| ...has a different answer column | `X = df.drop(columns=[...])`, `y = df[...]` | 3.4 |
| ...has different label words | `Series.map({...})` or `LabelEncoder` | 3.5 |
| ...needs only some rows (one country) | `df[df["Country"] == "..."]` | 3.6 |
| ...has numbers stored as text ("£5") | `.str.replace` + `pd.to_numeric` | 4.1 |
| ...has gaps | `fillna(median / mode)` or `dropna` | 4.1 |
| ...has text category columns | `pd.get_dummies` | 4.2 |
| ...has columns on very different scales | `StandardScaler` (Week 2 doesn't scale) | 4.3 |
| ...is transactions with no country or quantity | Change the basket-building chain | 4.4 |
| ...is fewer than 2,000 texts | `imdb.sample(min(2000, len(imdb)))` | 4.5 |
| ...has more than 2 classes | `f1_score(..., average="macro")`; the label `map` | 5.1, 5.5 |
| ...has a class with only 1 row | Remove `stratify` or that class | 5.1 |
| ...needs a different number of clusters | `k = ...` (Week 3, line 125) | 5.3 |
| ...gives DBSCAN one cluster or all noise | `DBSCAN(eps=...)` | 5.3 |
| ...gives no association rules | `apriori(min_support=...)` | 5.4 |
| ...gives charts with the wrong labels | `plt.xlabel` / `plt.ylabel` / `plt.title` text | 6 |
