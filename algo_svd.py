from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate

# Load the movielens-100k dataset
data = Dataset.load_builtin()

# Use the famous SVD algorithm.
# Built in algo: https://surprise.readthedocs.io/en/stable/prediction_algorithms_package.html
algo = SVD()

# Run 5-fold cross-validation and print results.
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)