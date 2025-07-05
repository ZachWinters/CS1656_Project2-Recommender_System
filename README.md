# Movie Recommender System
**Technologies:** Python, Pandas, NumPy, SciPy
(Euclidean/Manhattan/Cosine/Pearson metrics

## Project Overview
Implements a user-based collaborative filtering recommender system with four-different similarity metrics to predict movie ratings. The system analyzed user behavior patterns to generate personalized recommendations. Furthermore, evalutates prediction accuracy using RMSE and coverage metrics

## Key Features
* **Multiple Similarity Metrics**: Implements four distinct algorithms to calculate user similarity:
  - Euclidean Distance
  - Manhattan Distance
  - Cosine Similarity
  - Pearson Correlation
* **Top-K Neighbor Selection**: Predicts ratings by aggregating opinions from the most similar users (configurable K value)
* **Performance Evaluation**: Measures prediction quality using:
  * Root Mean Square Error (RMSE)
  * Coverage Ratio (percentage of predictable ratings)
  
## Core Components
1. **Data Processing**:
  * Handles CSV input files containing user-movie ratings
  * Uses Pandas DataFrames for efficient data manipulation
2. **Similarity Calculations**:
  * Leverages SciPy's optimized distance functions
  * Implements custom Pearson correlation computation
3. **Prediction Engine**:
  * Weighted average rating prediction from top-K similar users
  * Handles missing data (NaN values) gracefully
4. **Evaluation Framework**:
  * Comparative analysis of different similarity metrics
  * K-value optimization testing

How It Works:

Calculates user similarity weights from training data
Predicts ratings using weighted averages from top-k similar users
Evaluates performance via RMSE and coverage metrics
