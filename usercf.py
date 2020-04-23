import surprise
import pandas as pd
import numpy as np
import compress_pickle
from compress_pickle import dump, load

from surprise import Reader
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import KNNBaseline
from surprise import KNNWithMeans
from surprise import KNNBasic
from surprise import KNNWithZScore
from surprise import SVD
from surprise import BaselineOnly
from surprise import NMF
from surprise import CoClustering
from surprise.model_selection import GridSearchCV
from surprise.accuracy import rmse
from surprise import accuracy
from surprise.model_selection import train_test_split

from collections import defaultdict

ratings = pd.read_csv("D:/Downloads/BX-CSV-Dump/BX-Book-Ratings.csv", delimiter=";", encoding="latin1")
ratings.columns = ['user_id', 'ISBN', 'bookRating']

users = pd.read_csv('D:/Downloads/BX-CSV-Dump/BX-Users.csv', delimiter=";", encoding="latin1")
users.columns = ['user_id', 'location', 'age']

books = pd.read_csv('D:/Downloads/BX-CSV-Dump/BX_Books.csv', sep=';',encoding='latin-1')
books.columns=['ISBN', 'book_title' ,'book_author','year_of_publication', 'publisher', 'img_s', 'img_m', 'img_l']

# Pre-Processing functions:
def get_data_surp_scratch(books, users, ratings):
    data = pre_process_merge_pipeline(books, users, ratings)
    data_expl, data_impl = explicit_implicit_transform(data)
    data_surp = get_data_surp(data_expl)
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(data_surp[['user_id', 'ISBN', 'bookRating']], reader)
    # data_surp is a df
    # data is a surprise object of the Dataset module which is used for models.
    return data_surp, data
def get_data_surp(data_expl):
    x = data_expl.groupby(['user_id'])
    u = []
    # Users who have rated more than 50 books.
    for i, j in x:
        if j.shape[0] > 50:
            u.append(i)
    data_surp = data_expl[data_expl['user_id'].isin(u)]
    data_surp = data_surp.sort_values('user_id').reset_index(drop=True)

    # Books which have more than 5 number of ratings
    y = data_surp.groupby(['ISBN'])
    b = []
    for i, j in y:
        if j.shape[0] > 10:
            b.append(i)
    data_surp = data_surp[data_surp['ISBN'].isin(b)]
    data_surp = data_surp.sort_values('user_id').reset_index(drop=True)

    return data_surp
def explicit_implicit_transform(data):
    # returns two dataframes, one with explicit rating and one with implicit rating.
    data_expl = data[data.bookRating != 0]
    data_impl = data[data.bookRating == 0]
    return data_expl, data_impl
def pre_process_merge_pipeline(books, users, ratings):
    books = pre_process_books(books)
    users = pre_process_users(users)
    ratings = pre_process_rating(ratings)

    # Removing duplicates in users
    users = users[users.user_id.isin(ratings.user_id)]

    ratings_books = pd.merge(ratings, books, on='ISBN')

    # Replacing not defined publishers and authors.
    ratings_books.book_author.fillna('unknown', inplace=True)
    ratings_books.publisher.fillna('unknown', inplace=True)

    data = pd.merge(ratings_books, users, on='user_id')
    return data
def pre_process_users(users):
    # Not much handling for user_id
    users.user_id = users.user_id.astype(int)

    # Too many na values for age, so created a normal distribution for na values.
    users.age = users.age.astype(float)
    users.loc[(users.age > 99) | (users.age < 5), 'age'] = np.nan
    # create a normal disgtribution pd.Series to fill Nan values with because you cannot just replace it with mean
    # as there are a large no. of nan values.
    rand_dist = pd.Series(np.random.normal(loc=users.age.mean(),
                                           scale=users.age.std(),
                                           size=users.user_id[users.age.isna()].count()))
    # Eliminating the negative values in the random distribution
    age_series = np.abs(rand_dist)
    # sorting users Df so as NaN values in age to be first
    # reset index to match with index of age_series
    # Then use fillna()
    users = users.sort_values('age', na_position='first').reset_index(drop=True)
    users.age.fillna(age_series, inplace=True)
    # replace values < 5 with the mean(). Round values and convert them to int.
    users.loc[users.age < 5, 'age'] = users.age.mean()
    users.age = users.age.round().astype(int)
    # Sort users based on user_id so as to be the same as before
    users = users.sort_values('user_id').reset_index(drop=True)

    # Dropping users location because
    users.drop('location', axis=1, inplace=True)

    return users
def pre_process_books(books):
    # Dropping unneccesary
    books.drop(['img_s', 'img_m', 'img_l'], axis=1, inplace=True)
    books.year_of_publication = books.year_of_publication.astype(int)

    # Replacing na values
    books.loc[187701, 'book_author'] = "n/a"
    books.loc[[128897, 129044], 'publisher'] = "NovelBooks, Inc"
    books.loc[(books.year_of_publication > 2010) | (books.year_of_publication < 1500), 'year_of_publication'] = np.nan
    books.year_of_publication.fillna(round(books.year_of_publication.mean()), inplace=True)

    # Changing dtype to save memory
    books.year_of_publication = books.year_of_publication.astype(int)

    ## REMOVING DUPLICATE VALUES
    books = books.drop_duplicates(['book_title', 'book_author'])

    return books
def pre_process_rating(ratings):
    ratings = ratings[ratings.ISBN.isin(books.ISBN)]

    return ratings

data_surp,data = get_data_surp_scratch(books,users,ratings)
data_surp.shape
data

users_list=list(data_surp.user_id.unique())
len(users_list)
books_list=list(data_surp.ISBN.unique())
len(books_list)

# Conversion and History Functions
def get_book_title(ISBN):
    return (data_surp.loc[(data_surp.ISBN == str(ISBN)), 'book_title']).iloc[0]
def get_book_id(book_title):
    return (data_surp.loc[(data_surp.book_title == str(book_title)), 'ISBN'].iloc[0])
def get_rated_books_list(user_id):
    book_list = []
    x = (data_surp.loc[(data_surp.user_id == int(user_id)), 'ISBN'].tolist())
    for i in range(0, len(x)):
        book_list.append(get_book_title(x[i]))
    return book_list
get_book_title('0440168724')
# TOP N RECOMMENDATIONS FOR A NEW USER
def topn_recommendations_new_user(n,data_surp):
  ratings_count = pd.DataFrame(data_surp.groupby(['ISBN'])['bookRating'].sum())
  topn = ratings_count.sort_values('bookRating', ascending = False).head(n)
  print("Following books are recommended:")
  topn=topn.merge(data_surp, left_index = True, right_on = 'ISBN')
  return list(topn.book_title.unique())

# USER BASED CF FUNCTION
def get_n_recommendations_for_user(n, user_id):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in uscf_predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)

    all_pred = top_n

    for uid, user_ratings in all_pred.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        all_pred[uid] = user_ratings[:n]

    tmp = pd.DataFrame.from_dict(all_pred)
    tmp_transpose = tmp.transpose()

    results = tmp_transpose.loc[user_id]
    recommended_book_ids = []
    recommended_book_titles = []

    for x in range(0, n):
        recommended_book_ids.append(results[x][0])

    for i in range(0, len(recommended_book_ids)):
        recommended_book_titles.append(get_book_title(recommended_book_ids[i]))

    return recommended_book_titles

# ITEM BASED CF FUNCTION
def get_n_similar_books(n,book,itcf):
  # Retrieve inner id of the movie Toy Story
  book_raw_id = get_book_id(book)
  book_inner_id = itcf.trainset.to_inner_iid(book_raw_id)

  # Retrieve inner ids of the nearest neighbors of Toy Story.
  book_neighbors = itcf.get_neighbors(book_inner_id, k=n)

  # Convert inner ids of the neighbors into names.
  book_neighbors = (itcf.trainset.to_raw_iid(inner_id)
                            for inner_id in book_neighbors)
  book_neighbors = (get_book_title(rid)
                            for rid in book_neighbors)
  print()
  print('The '+str(n)+' nearest neighbors of '+str(book)+' are:')
  for book in book_neighbors:
    print(book)


# # USER MODEL:
# epochs = 20
# lr_value = 0.005
# reg_value = 0.15
#
# trainset = data.build_full_trainset()   #Build on entire data set
# uscf_algo = SVD(n_epochs=epochs, lr_all=lr_value, reg_all=reg_value)
# uscf_algo.fit(trainset)
#
# # Predict ratings for all pairs (u, i) that are NOT in the training set.
# uscf_testset = trainset.build_anti_testset()
#
# #Predicting the ratings for testset
# uscf_predictions = uscf_algo.test(uscf_testset)
# accuracy.rmse(uscf_predictions)

uscf_predictions=load('usercf.pkl')

get_n_recommendations_for_user(10,1424)
get_rated_books_list(1424)

# # ITEM-BASED MODEL
# ## First, train the algortihm to compute the similarities between items
# trainset = data.build_full_trainset()
# sim_options = {'name': 'pearson_baseline', 'user_based': False,
#                'shrinkage':10 }
#
# bsl_options={'bsl_options': {'method': ['sgd'],
#                               'reg': [1]}}
# itcf_algo = KNNBaseline(sim_options=sim_options,bsl_options=bsl_options)
# itcf=itcf_algo.fit(trainset)
# dump(itcf,'itemcf.pkl')



