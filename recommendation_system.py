import numpy as np
from numpy.linalg import norm
import math
 
def read_matrix(fname, delim):
    data = []
    with open(fname, "r") as f:
        for line in f:
            nstrs = line.split(delim)
            nums = [int(n) for n in nstrs]
            data.append(nums)
    data = np.array(data)
    return data

def getIUF(train_data):
    user_count = np.sum( np.logical_and(np.ones(train_data.shape), train_data), axis=0)
    user_count[user_count[:] == 0] = 1 # Minimum count of 1
    return np.log(200 / user_count)


def cosine_similarity(train_data, active_ratings, do_iuf=False):
    if do_iuf:
        iuf = getIUF(train_data)
        train_data *= iuf
        active_ratings *= iuf

    return np.dot(train_data, active_ratings) / (norm(train_data, axis=1)*norm(active_ratings))

def item_adj_cos_sim(train_data, movie_id):
    train_data = np.array(train_data, dtype=np.float64)

    for i, user_ratings in enumerate(train_data):
        avg = np.mean(user_ratings)
        for j, r in enumerate(user_ratings):
            if r != 0:
                train_data[i, j] -= avg

    similarities = np.zeros(train_data.shape[1])

    ratings_j = train_data[:, movie_id]

    for i in range(train_data.shape[1]):
        ratings_i = train_data[:, i]
        similarities[i] = np.dot(ratings_i, ratings_j)
        similarities[i] /= np.sqrt(np.sum(np.power(ratings_i, 2)))
        similarities[i] /= np.sqrt(np.sum(np.power(ratings_j, 2)))
        if (np.isnan(similarities[i])):
            similarities[i] = 0.0

    return similarities


def pearson_similarity(train_data, active_ratings, do_iuf=False):
    train_data = np.array(train_data, dtype=np.float64)
    active_ratings = np.array(active_ratings, dtype=np.float64)

    if do_iuf:
        iuf = getIUF(train_data)
        train_data *= iuf
        active_ratings *= iuf

    ra = np.mean(active_ratings)
    for i, r in enumerate(active_ratings):
        if r != 0:
            active_ratings[i] -= ra

    similarities = np.zeros(train_data.shape[0])

    for i, user_ratings in enumerate(train_data):
        ru = np.mean(user_ratings)
        for j, r in enumerate(user_ratings):
            if r != 0:
                train_data[i, j] -= ru

    similarities = np.dot(train_data, active_ratings)
    similarities /= np.sqrt(np.sum(np.power(active_ratings, 2)))
    similarities /= np.sqrt(np.sum(np.power(train_data, 2), axis=1))
    #print(similarities)
    return similarities

def knearestneighbor(similarities, k):
    if k > similarities.size:
        k = similarities.size
     
    # k Nearest Neighbor
    invk = similarities.size - k
    topk = np.argpartition(similarities, invk)[invk:]
         
    ksims = similarities[topk]
    return ksims, topk
 
def basic_collab_filter(train_data, user_ratings, movie_id, k, calc_similarities=pearson_similarity, do_iuf=False, do_ca=False, do_secret=False):
    # remove users who haven't rated the movie
    train_data = train_data[train_data[:, movie_id - 1] != 0]
 
    similarities = calc_similarities(train_data, user_ratings, do_iuf)

    if do_secret:
        similarities *= np.log(np.sum(np.logical_and(train_data, user_ratings)))
 
    if do_ca:
        similarities = similarities * np.power(np.abs(similarities), 1.5)

    ksims, topk = knearestneighbor(similarities, k)
    
    if np.array_equal(ksims, np.zeros(ksims.size)):
        return 3
 
    result = np.dot(ksims, train_data[topk, movie_id - 1]) / np.sum(ksims)
 
    return result

def item_collab_filter(train_data, user_ratings, movie_id, k, calc_similarities=item_adj_cos_sim):
    # remove movies user hasn't rated
    train_data = train_data[:, np.nonzero(user_ratings)[0]]
    # remove nonzero entries form user data
    user_ratings = user_ratings[np.nonzero(user_ratings)[0]]
    
    similarities = calc_similarities(train_data, movie_id)

    result = np.dot(similarities, user_ratings) / np.sum(similarities)

    return result
 
def get_user_ratings(user_data):
    user_ratings = np.zeros(1000)
    for row in user_data:
        user_ratings[row[1] - 1] = row[2]
    return user_ratings
 
def write_result(m, calc_similarities, k, do_iuf=False, do_ca=False, collab_filter=basic_collab_filter):
    resultf = open("result" + str(m) + ".txt", "w")
 
    train_data = read_matrix("train.txt", "\t")
    test_data = read_matrix("test" + str(m) + ".txt", " ")
 
    i = 0
    guesses = []
    while (i < test_data.shape[0]):
        user_data = test_data[i:i+m, :]
        user_ratings = get_user_ratings(user_data)
        i += m
         
        while (True):
            if i >= test_data.shape[0] or test_data[i][2] != 0:
                break
 
            guess = 0
            if collab_filter == basic_collab_filter:
                guess = collab_filter(train_data, user_ratings, test_data[i][1], k, calc_similarities)
            elif collab_filter == item_collab_filter:
                guess = collab_filter(train_data, similarities, movie_id, k)
            
            if (np.isnan(guess)):
                print(i)
            guesses.append(guess)
            resultf.write("{} {} {} \n".format(test_data[i][0], test_data[i][1], int(round(guess))))
            i += 1
 
    resultf.close()
    print(np.mean(np.array(guesses)))

def test(calc_similarities, k, do_iuf=False, do_ca=False, collab_filter=basic_collab_filter):
    data = read_matrix("train.txt", "\t")
    train_data = data[:180, :]
    test_data = data[180:, :]

    errors = []

    for user in test_data:
        for i, movie_id in enumerate(user):
            if (user[i] == 0):
                continue

            sim_user = np.copy(user)
            sim_user[i] = 0
            guess = 0
            if collab_filter == basic_collab_filter:
                guess = basic_collab_filter(train_data, sim_user, movie_id, k, calc_similarities, do_iuf, do_ca)
            elif collab_filter == item_collab_filter:
                guess = item_collab_filter(train_data, sim_user, movie_id, k)
            errors.append((guess - user[i])**2)

    errors = np.array(errors)
    rmse = math.sqrt(np.sum(errors) / errors.size)
    return rmse

k = 20

# print(test(cosine_similarity, k))
# print(test(pearson_similarity, k))
# print(test(pearson_similarity, k, do_iuf=True))
# print(test(pearson_similarity, k, do_ca=True))
# print(test(pearson_similarity, k, do_ca=True, do_iuf=True))
# print(test(item_adj_cos_sim, k, collab_filter=item_collab_filter))

for m in [5, 10, 20]:
    write_result(m, cosine_similarity, 20)