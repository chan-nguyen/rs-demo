from surprise import KNNBasic
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import KFold

# Load the movielens-100k dataset
data = Dataset.load_builtin('ml-100k')

# define a cross-validation iterator
kf = KFold(n_splits=2)

sim_options = {
    'name': 'cosine', # pearson, cosine, msd
    'user_based': 'True'
}

algo = KNNBasic(sim_options = sim_options)

# cross validation https://scikit-learn.org/stable/modules/cross_validation.html
for trainset, testset in kf.split(data):
    # train and test algorithm.
    algo.fit(trainset)
    predictions = algo.test(testset)

    # Compute and print Root Mean Squared Error
    accuracy.mae(predictions, verbose=True)