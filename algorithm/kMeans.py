import helper as hp
import numpy as np


def findRemotestIndex(data_mat, idx_list):
    remotest_idx = 0
    max_dist = 0
    for i in range(data_mat.shape[0]):
        min_dist = np.inf
        for j in range(len(idx_list)):
            if i == idx_list[j]:
                min_dist = -np.inf
                continue
            cur_dist = hp.angDist(data_mat[i, :], data_mat[idx_list[j]])
            if cur_dist < min_dist:
                min_dist = cur_dist
        if min_dist > max_dist:
            max_dist = min_dist
            remotest_idx = i
    return remotest_idx


def remotestCenter(data_mat, k):
    n = data_mat.shape[1]
    centroids = np.mat(np.zeros((k, n)))
    ini_idx = np.random.randint(0, data_mat.shape[0])
    arr = [ini_idx]
    while len(arr) < k:
        remotest_idx = findRemotestIndex(data_mat, arr)
        arr.append(remotest_idx)
    for i in range(len(arr)):
        centroids[i, :] = data_mat[arr[i], :]
    return centroids


def redefinedCenter(data_mat, cent_idx):
    centroids = np.mat(np.zeros((len(cent_idx), data_mat.shape[1])))
    for i in range(len(cent_idx)):
        centroids[i, :] = data_mat[cent_idx[i], :]
    return centroids


def randCenter(data_mat, k):
    n = data_mat.shape[1]
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        min_j = np.min(data_mat[:, j])
        range_j = np.float(np.max(data_mat[:, j]) - min_j)
        centroids[:, j] = min_j + range_j * np.random.rand(k, 1)
    return centroids


def angKmeans(data_set, k, create_cent=randCenter, calc_mean=hp.orientationMean, max_iterate=50,
              min_error=0.0):
    copy_set = data_set.copy()
    m = copy_set.shape[0]
    cluster_condition = np.zeros((m, 2))
    centroids = create_cent(copy_set, k)
    ini_centroids = centroids.copy()
    cluster_changed = True
    iterate_count = 0
    old_error = 0
    while cluster_changed:
        iterate_count = iterate_count + 1
        cluster_changed = False
        for i in range(m):
            min_dist = np.inf
            min_index = -1
            for j in range(k):
                dist_ji = hp.angDist(centroids[j, :], copy_set[i, :])
                if dist_ji < min_dist:
                    min_dist = dist_ji
                    min_index = j
            if cluster_condition[i, 0] != min_index:
                cluster_changed = True
                cluster_condition[i, :] = min_index, min_dist ** 2
        max_dist = -np.inf
        max_index = -1
        for i in range(m):
            if cluster_condition[i, 1] > max_dist:
                max_dist = cluster_condition[i, 1]
                max_index = i
        for cent in range(k):
            pts_cluster = copy_set[np.nonzero(cluster_condition[:, 0] == cent)]
            if pts_cluster.shape[0] == 0:
                centroids[cent, :] = copy_set[max_index, :]
                print("empty slice happened")
                cluster_changed = True
            else:
                centroids[cent, :] = calc_mean(pts_cluster)
        if iterate_count >= max_iterate:
            cluster_changed = False
        if np.abs(np.sum(cluster_condition[:, 1]) / m - old_error) <= min_error:
            cluster_changed = False
        old_error = np.sum(cluster_condition[:, 1]) / m
        print("iterate round ", iterate_count, ", average error = ", np.sum(cluster_condition[:, 1]) / m)
    return ini_centroids, centroids, cluster_condition
