# -*- coding:utf-8 -*-

import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

# fetch data and format it
data = fetch_movielens(data_home='data/', min_rating=4.0)

# print training and testing data
print repr(data['train'])
print repr(data['test'])

# create model
model = LightFM(loss='warp')
# train model
model.fit(data['train'], epochs=30)

def sample_recommendation(model, data, user_ids):

    # number of users and movies in training data
    n_users, n_items = data['train'].shape

    # generate recommendations for each user we input
    for user_id in user_ids:
        # movie they already like
        know_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        # movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))

        # remove the movie that user has seen
        unseen_mask = np.in1d(scores, data['train'].tocsr()[user_id].toarray(), assume_unique=True, invert=True)
        unseen_scores = scores[unseen_mask]

        # rank them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-unseen_scores)]

        # print the result
        print 'User {}'.format(user_id)
        print "     Known positives:"

        for x in know_positives[:3]:
            print "              {}".format(x)

        print "     Recommended:"

        for x in top_items[:3]:
            print "              {}".format(x)

sample_recommendation(model, data, [3, 25, 450])


