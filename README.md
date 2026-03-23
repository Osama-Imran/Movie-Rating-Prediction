# Movie Rating Prediction

## Overview
This project builds a machine learning model to predict how a user might rate a movie they haven't seen yet. Using the MovieLens dataset, a Random Forest Regressor is trained on user and movie features to estimate ratings on a scale of 1 to 5.

---

## Dataset
**Source:** MovieLens Latest Small Dataset (GroupLens)
**Files Used:** `movies.csv`, `ratings.csv`
**Total Ratings:** 100,836
**Total Movies:** 9,742

| File | Columns |
|---|---|
| ratings.csv | userId, movieId, rating, timestamp |
| movies.csv | movieId, title, genres |

---

## Project Steps

**1. Data Loading & Exploration**
Loaded both CSV files and explored structure, shape, and null values. No missing values were found in either file.

**2. Merging Datasets**
Merged `ratings.csv` and `movies.csv` on `movieId` to create a single unified dataframe with all relevant information per rating.

**3. Preprocessing**
- Dropped `timestamp` — not useful for predicting ratings
- Dropped `title` and `genres` after feature extraction

**4. Feature Engineering**
- Extracted `year` from movie title using regex (e.g. `Toy Story (1995)` → `1995`)
- Extracted `main_genre` as the first listed genre from the pipe-separated genres column
- Label Encoded `main_genre` to convert text to numeric values

**5. Model Training**
Trained a `RandomForestRegressor` with 100 decision trees on 80% of the data, using userId, movieId, year, and main_genre as features.

**6. Evaluation**
Evaluated on the remaining 20% test data using regression metrics — RMSE and MAE.

---

## Results

| Metric | Score |
|---|---|
| **RMSE** | 0.9914 |
| **MAE** | 0.7600 |

RMSE below 1.0 means predictions are off by less than 1 star on average — considered a good result for movie rating prediction.

---

## Feature Importance

| Feature | Importance |
|---|---|
| movieId | 0.44 |
| userId | 0.35 |
| year | 0.13 |
| main_genre | 0.08 |

---

## Key Insights
- **movieId is the strongest predictor** — some movies are universally loved or disliked regardless of who watches them
- **userId matters almost as much** — individual rating behavior varies greatly, some users are generous raters, others are harsh critics
- **Year has moderate influence** — classic movies tend to be rated differently than newer releases
- **Genre matters least** — personal preference overrides genre bias, proving that ratings are deeply individual
- **Extreme ratings like 0.5 are hardest to predict** — they are rare in the dataset so the model has limited examples to learn from

---

## Libraries Used
- `pandas` — data manipulation
- `numpy` — numerical operations
- `matplotlib` — visualization
- `scikit-learn` — model training and evaluation
