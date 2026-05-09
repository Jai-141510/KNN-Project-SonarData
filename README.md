# Sonar Object Detection — KNN Classifier

> Can a machine tell the difference between a mine and a rock using only sound? Turns out, mostly yes.

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat&logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-red?style=flat&logo=jupyter)

---

## 📋 Dataset

| Property | Detail |
|---|---|
| Source | UCI Sonar Dataset |
| Samples | 208 |
| Features | 60 sonar frequency responses |
| Target | M = Mine, R = Rock |

Each row is a sonar reading 60 numbers representing how strongly different sound frequencies bounced back off an object. The model's job is to decide: mine or rock?

---

## 🔍 What Was Done

**EDA** : No missing values, no type issues. All 60 features were already floats between 0 and 1. Clean dataset straight out of the box.

**Label encoding** : Converted M → 1 and R → 0 before modelling. Mines are the positive class since they're what we care about detecting.

**Pipeline with operations variable** : Rather than writing the pipeline directly, defined an `operations` list first (StandardScaler → KNeighborsClassifier) and passed it into `Pipeline(operations)`. Keeps the code readable and the steps easy to swap.

**Baseline** : 5-fold cross validation before any tuning gave 79% accuracy and 91% recall. Decent starting point, though an 8% train/test gap suggested mild overfitting with the default K.

**GridSearchCV** — tuned three parameters together: number of neighbours (odd values 3–21), weighting strategy (uniform vs distance), and distance metric (euclidean vs manhattan). First two runs flagged K=1 and K=2 as winners, both red flags. K=1 memorises the training data; K=2 causes tie votes in binary classification. Reran with odd values starting at 3.

**Final model** : K = 3, uniform weights, euclidean distance.

---

## 📊 Results

| Metric | Score |
|---|---|
| Accuracy | 86% |
| Mine Recall | 82% |
| Mine Precision | 90% |
| F1 | 86% |

When the model says mine, it's right 90% of the time. It misses 2 out of every 11 actual mines, which in a real naval context, would need a secondary verification step before clearing anything as safe.

---

## ⚠️ Key Lessons

**KNN needs scaling** : Without StandardScaler inside the pipeline, results would be meaningless. Distance based models are sensitive to feature magnitude.

**Odd K values for binary classification** : Even K values risk tie votes. Always start at 3.

**Small K ≠ good K** : GridSearch wanted K = 1, which is just memorisation. Domain knowledge and sanity checks matter as much as the metric.

---

## 🛠️ Stack

`Python` &nbsp;|&nbsp; `scikit-learn` &nbsp;|&nbsp; `pandas` &nbsp;|&nbsp; `matplotlib`

---

## 📁 Files

```
├── KNN-Sonar-Project.ipynb   # Full notebook
└── sonar_all-data.csv        # Dataset
```
