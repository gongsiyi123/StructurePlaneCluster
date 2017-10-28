import helper as hp
import numpy as np


def calcSimilarity(data_set, p_mode=0):
    similarity = np.mat(np.zeros((data_set.shape[0], data_set.shape[0])))
    for i in range(data_set.shape[0]):
        temp = np.mat(np.zeros((1, data_set.shape[0])))
        for j in range(data_set.shape[0]):
            s = -1 * hp.angDist(data_set[i, :], data_set[j, :])
            temp[0, j] = s
        similarity[i, :] = temp[0, :]
    if p_mode == 0:
        p = np.median(similarity.A)
    elif p_mode == -1:
        p = np.min(similarity)
    elif p_mode == 1:
        p = np.max(similarity)
    else:
        p = np.median(similarity)
    for i in range(similarity.shape[0]):
        similarity[i, i] = p
    return similarity


def maxWithout(data_arr, k, axis=1):
    row_without_k = np.delete(data_arr, k, axis=axis)
    return np.max(row_without_k)


def iterR(r, a, s, lam):
    n = r.shape[0]
    for i in range(n):
        for k in range(n):
            old_r = r[i, k]
            a_s = a[i, :] + s[i, :]
            r[i, k] = s[i, k] - maxWithout(a_s, k)
            r[i, k] = (1 - lam) * r[i, k] + lam * old_r
    return r


def iterA(r, a, lam):
    n = a.shape[0]
    for i in range(n):
        for k in range(n):
            old_a = a[i, k]
            if i != k:
                sum_no_i_k = 0
                for j in range(n):
                    if j == i or j == k:
                        continue
                    r_j = r[j, k]
                    if r_j < 0:
                        r_j = 0
                    sum_no_i_k = sum_no_i_k + r_j
                a[i, k] = r[k, k] + sum_no_i_k
                if a[i, k] > 0:
                    a[i, k] = 0
            else:
                sum_no_k = 0
                for j in range(n):
                    if j == k:
                        continue
                    r_j = r[j, k]
                    if r_j < 0:
                        r_j = 0
                    sum_no_k = sum_no_k + r_j
                a[k, k] = sum_no_k
                if a[k, k] < 0:
                    a[k, k] = 0
            a[i, k] = (1 - lam) * a[i, k] + lam * old_a
    return a


def calcCentroids(s, r, a, lam, max_iter, ):
    cur_iter = 0
    clus_idx = []
    while True:
        r = iterR(r, a, s, lam)
        a = iterA(r, a, lam)
        cur_iter = cur_iter + 1
        print("round ", cur_iter, " : ")
        if cur_iter >= max_iter:
            break
    r_a = r + a
    for k in range(s.shape[0]):
        max_val = r_a[k, 0]
        max_idx = 0
        for j in range(s.shape[1]):
            if r_a[k, j] > max_val:
                max_val = r_a[k, j]
                max_idx = j
        if max_idx not in clus_idx:
            clus_idx.append(max_idx)
    return clus_idx


def generateCluster(data_set, clus_idx):
    clus_list = []
    for i in range(data_set.shape[0]):
        temp = []
        for j in range(len(clus_idx)):
            d = -1 * hp.angDist(data_set[i, :], data_set[clus_idx[j]])
            temp.append(d)
        c = clus_idx[temp.index(np.max(temp))]
        clus_list.append(c)
    return clus_list


def affinityPropagation(data_set, lam=0.5, max_iter=50):
    copy_set = data_set.copy()
    r = np.mat(np.zeros((copy_set.shape[0], copy_set.shape[0])))
    a = np.mat(np.zeros((copy_set.shape[0], copy_set.shape[0])))
    s = calcSimilarity(copy_set, 0)
    print("*************** similarity finished ***************")
    clus_idx = calcCentroids(s, r, a, lam, max_iter)
    print("*************** centroids finished ***************")
    print(clus_idx)
    clus_list = generateCluster(data_set, clus_idx)
    centroids = np.mat(np.zeros((len(clus_idx), copy_set.shape[1])))
    for i in range(centroids.shape[0]):
        centroids[i, :] = copy_set[clus_idx[i], :]
    return centroids, clus_list
