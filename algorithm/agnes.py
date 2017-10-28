import numpy as np
import helper as hp


def dist_min(data_set_1, data_set_2):
    min_dis = 1
    for i in range(data_set_1.shape[0]):
        for j in range(data_set_2.shape[0]):
            dis_ij = hp.angDist(data_set_1[i, :], data_set_2[j, :])
            if dis_ij < min_dis:
                min_dis = dis_ij
    return min_dis


def dist_max(data_set_1, data_set_2):
    max_dis = 1
    for i in range(data_set_1.shape[0]):
        for j in range(data_set_2.shape[0]):
            dis_ij = hp.angDist(data_set_1[i, :], data_set_2[j, :])
            if dis_ij > max_dis:
                max_dis = dis_ij
    return max_dis


def dist_mean(data_set_1, data_set_2):
    sum_dis = 0
    for i in range(data_set_1.shape[0]):
        for j in range(data_set_2.shape[0]):
            sum_dis = sum_dis + hp.angDist((data_set_1[i, :], data_set_2[j, :]))
    return sum_dis / (data_set_1.shape[0] * data_set_2.shape[0])


def agnes(data_set, k, dist_calc=dist_min):
    C = []
    M = np.mat(np.zeros((data_set.shape[0], data_set.shape[0])))
    for j in range(data_set.shape[0]):
        C.append(data_set[j, :])
    for i in range(data_set.shape[0]):
        for j in range(i + 1, data_set.shape[0]):
            M[i, j] = dist_calc(C[i], C[j])
            M[j, i] = M[i, j]
    q = data_set.shape[0]
    print("*************** initialize finished ***************")
    count = 1
    while q > k:
        min_dis = 1
        min_i = 0
        min_j = 0
        for i in range(len(C)):
            for j in range(i + 1, len(C)):
                dis = dist_calc(C[i], C[j])
                if dis < min_dis:
                    min_dis = dis
                    min_i = i
                    min_j = j
        C[min_i] = np.row_stack((C[min_i], C[min_j]))
        for j in range(min_j, len(C) - 1):
            C[j] = C[j + 1]
        C.pop()
        M = np.delete(M, min_j, 0)
        M = np.delete(M, min_j, 1)
        for j in range(q - 1):
            M[min_i, j] = dist_calc(C[min_i], C[j])
            M[j, min_i] = M[min_i, j]
        q = q - 1
        print("layer ", str(count), " finished, now q = " + str(q))
        count = count + 1
    idx = []
    for i in range(data_set.shape[0]):
        idx.append(0)
    for i in range(data_set.shape[0]):
        for m in range(len(C)):
            for n in range(C[m].shape[0]):
                if (data_set[i, :] == C[m][n, :]).all():
                    idx[i] = m
    centroids = np.mat(np.zeros((len(C), 3)))
    for i in range(len(C)):
        centroids[i] = hp.orientationMean(C[i])
    return centroids, idx
