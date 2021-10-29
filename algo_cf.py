#https://github.com/grahamjenson/list_of_recommender_systems
from surprise import Dataset, KNNBasic
from surprise.model_selection import cross_validate

# https://grouplens.org/datasets/movielens/100k/
# https://www.kaggle.com/prajitdatta/movielens-100k-dataset
dataset = Dataset.load_builtin('ml-100k')

ratings = dataset.raw_ratings

print("Number of rating instances: ", len(ratings))
print("Number of unique users: ", len(set([x[0] for x in ratings])))
print("Number of unique items: ", len(set([x[1] for x in ratings])))

sim_options = {
    'name': 'cosine', # pearson, cosine, msd
    'user_based': 'False'
}

clf = KNNBasic(sim_options = sim_options)

# cross validation https://scikit-learn.org/stable/modules/cross_validation.html
cross_validate(clf, dataset, measures=['MAE'], cv=5, verbose=True)