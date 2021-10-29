import pickle
import os
from typing import Counter

import pandas as pd

from surprise import KNNBasic
from surprise import Dataset                                                     
from surprise import Reader                                                      
from surprise import dump
from surprise.accuracy import rmse
from surprise.model_selection.split import PredefinedKFold

# We will train and test on the u1.base and u1.test files of the movielens-100k dataset.

def load_dataset():
    # Now, let's load the dataset
    train_file = os.path.expanduser('~') + '/.surprise_data/ml-100k/ml-100k/u1.base'
    test_file = os.path.expanduser('~') + '/.surprise_data/ml-100k/ml-100k/u1.test'
    data = Dataset.load_from_folds([(train_file, test_file)], Reader('ml-100k'))

                    
    # We'll use a basic nearest neighbor approach, where similarities are computed
    # between users.
    algo = KNNBasic()  

    pkf = PredefinedKFold()

    trainset, testset = list(pkf.split(data))[0]
    algo.fit(trainset)                             
    predictions = algo.test(testset)
    # rmse(predictions)
    # dump.dump('./dump_file', predictions, algo)
    
    return predictions, algo

# The dump has been saved and we can now use it whenever we want.
# Let's load it and see what we can do
predictions, algo = load_dataset()
# predictions, algo = dump.load('./dump_file')
trainset = algo.trainset

# print('algo: {0}, k = {1}, min_k = {2}'.format(algo.__class__.__name__, algo.k, algo.min_k))

# Let's build a pandas dataframe with all the predictions

df = pd.DataFrame(predictions, columns=['uid', 'iid', 'rui', 'est', 'details']) 

print(df)

# df['err'] = abs(df.est - df.rui)

# best_predictions = df.sort_values(by='err')[:10]
# worst_predictions = df.sort_values(by='err')[-10:]

# # print(best_predictions)

# # print(worst_predictions)

# # Predict
# uid = str(196)  # raw user id (as in the ratings file). They are **strings**!
# iid = str(306)  # raw item id (as in the ratings file). They are **strings**!

# # get a prediction for specific users and items.
# pred = algo.predict(uid, iid, r_ui=4, verbose=True)


# recommend item for user
def recommend_item_for_user(uid):
    user_prediction = df.loc[df['uid'] == uid]
    top_10_prediction = user_prediction.sort_values('est', ascending=False)[0:10]
    return top_10_prediction['iid'].values

rec = recommend_item_for_user(str(196))
print (rec)

# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.style.use('ggplot')

# counter = Counter([r for (_, r) in trainset.ir[trainset.to_inner_iid('306')]])
# pd.DataFrame.from_dict(counter, orient='index').plot(kind='bar', legend=False)
# plt.xlabel('Rating value')
# plt.ylabel('Number of users')
# plt.title('Number of users having rated item 306')

# plt.show()
