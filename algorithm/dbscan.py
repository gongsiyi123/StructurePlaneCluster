import helper as hp
import numpy as np


def neighbour(data_set, data, eps):
    nb_set = data_set.copy()
    for i in range(data_set.shape[0] - 1, -1, -1):
        if hp.angDist(data_set[i, :], data) > eps:
            nb_set = np.delete(nb_set, i, axis=0)
    return nb_set


def removeRows(data_set, sub_set):
    removed_set = data_set.copy()
    for i in range(sub_set.shape[0]):
        for j in range(removed_set.shape[0]):
            t = removed_set[j, :] == sub_set[i, :]
            if np.all(t):
                removed_set = np.delete(removed_set, j, axis=0)
                break
    return removed_set


def containRow(data_set, data_row):
    for i in range(data_set.shape[0]):
        t = data_set[i, :] == data_row
        if np.all(t):
            return True
    return False


def intersectionMatrix(data_set_1, data_set_2):
    inter_set = np.mat(np.zeros((0, data_set_1.shape[1])))
    for i in range(data_set_1.shape[0] - 1, -1, -1):
        for j in range(data_set_2.shape[0]):
            t = data_set_2[j, :] == data_set_1[i, :]
            if np.all(t):
                inter_set = np.row_stack((inter_set, data_set_1[i, :]))
                break
    return inter_set


def dbscan(data_set, eps, min_pts):
    kernel = data_set.copy()
    for i in range(data_set.shape[0] - 1, -1, -1):
        print("calculate kernel round ", i)
        nb = neighbour(data_set, data_set[i, :], eps)
        if nb.shape[0] < min_pts:
            kernel = np.delete(kernel, i, axis=0)
    cks = np.mat(np.zeros((0, data_set.shape[1] + 1)))
    k = 0
    non_visited = data_set.copy()
    while kernel.shape[0] > 0:
        non_visited_old = non_visited.copy()
        if kernel.shape[0] <= 1:
            idx = 0
        else:
            idx = np.random.randint(0, kernel.shape[0] - 1)
        Q = kernel[idx, :]
        non_visited = removeRows(non_visited, Q)
        m = 0
        while Q.shape[0] > 0:
            m = m + 1
            print("finding connect round ", m)
            q = Q[0, :]
            Q = np.delete(Q, 0, axis=0)
            if containRow(kernel, q):
                nb_q = neighbour(data_set, q, eps)
                delta = intersectionMatrix(nb_q, non_visited)
                Q = np.row_stack((Q, delta))
                non_visited = removeRows(non_visited, delta)
                print(delta.shape[0], " , ", non_visited.shape[0], " , ", nb_q.shape[0])
        ck = removeRows(non_visited_old, non_visited)
        kernel = removeRows(kernel, ck)
        mark = np.mat(np.zeros((ck.shape[0], 1)))
        mark[:, 0] = k
        ck = np.c_[ck, mark]
        cks = np.row_stack((cks, ck))
        k = k + 1
        print("kernel set after ", k, " round calculate:")
    noise = removeRows(data_set, cks[:, 0:3])
    return cks, noise, k
