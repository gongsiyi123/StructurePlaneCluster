import math
import numpy as np


def loadDataSet(file_name):
    data_mat = []
    fr = open(file_name)
    for line in fr.readlines():
        cur_line = line.strip().split('\t')
        for i in range(len(cur_line)):
            cur_line[i] = float(cur_line[i])
        data_mat.append(cur_line)
    return data_mat


def degree2radian(data_mat, k):
    new_mat = data_mat.copy()
    for i in range(data_mat.shape[0]):
        if k < 0:
            for j in range(data_mat.shape[1]):
                new_mat[i, j] = math.radians(new_mat[i, j])
        else:
            new_mat[i, k] = math.radians(new_mat[i, k])
    return new_mat


def orientation2vector(data_mat):
    m = data_mat.shape[0]
    vector_mat = np.mat(np.zeros((m, 3)))
    vector_mat[:, 0] = np.multiply(np.cos(data_mat[:, 0]), np.sin(data_mat[:, 1]))
    vector_mat[:, 1] = np.multiply(np.sin(data_mat[:, 0]), np.sin(data_mat[:, 1]))
    vector_mat[:, 2] = np.cos(data_mat[:, 1])
    return vector_mat


def vector2orientation(data_mat):
    m = data_mat.shape[0]
    orientation_mat = np.mat(np.zeros((m, 2)))
    for i in range(m):
        x = data_mat[i, 0]
        y = data_mat[i, 1]
        z = data_mat[i, 2]
        if x == 0.0 and y <= 0.0:
            direction = 270.0
        elif x == 0.0 and y > 0.0:
            direction = 90.0
        else:
            direction = np.arctan(np.abs(y) / np.abs(x)) * 180.0 / np.pi
        if x > 0 and y > 0:
            direction = direction
        elif x < 0 and y > 0:
            direction = 180.0 - direction
        elif x < 0 and y < 0:
            direction = 180.0 + direction
        else:
            direction = 360.0 - direction
        angle = np.arccos(z) * 180.0 / np.pi
        if angle > 90.0:
            angle = 180.0 - angle
        orientation_mat[i, 0] = direction
        orientation_mat[i, 1] = angle
    return orientation_mat


def sieveDataIndex(data_mat, mean_mat, angle_limit):
    sieve_vector = np.zeros((data_mat.shape[0], 1))
    angle_limit_sin = np.power(np.sin(angle_limit * np.pi / 180), 2)
    for i in range(data_mat.shape[0]):
        for j in range(mean_mat.shape[0]):
            ang_dist = angDist(data_mat[i, :], mean_mat[j, :])
            if ang_dist <= angle_limit_sin:
                sieve_vector[i] = 1
    return sieve_vector


def sieveDataByIndex(data_mat, sieve_vector):
    sieve_mat = np.mat(np.zeros((data_mat.shape[0], data_mat.shape[1])))
    for i in range(data_mat.shape[0]):
        if sieve_vector[i] == 1:
            sieve_mat = np.row_stack((sieve_mat, data_mat[i, :]))
    if sieve_mat.shape[0] > 1:
        np.delete(sieve_mat, 1, 0)
    return sieve_mat


def angDist(vector1, vector2):
    cos_vector = np.multiply(vector1, vector2)
    sum_cos = np.sum(cos_vector)
    if sum_cos < 0:
        cos_vector = np.multiply(vector1, np.multiply(vector2, -1))
        sum_cos = np.sum(cos_vector)
    return 1 - np.power(sum_cos, 2)


def eucDist(vector1, vector2):
    cos_vector = np.multiply(vector1, vector2)
    sum_cos = np.sum(cos_vector)
    if sum_cos < 0:
        vector2 = np.multiply(vector2, -1)
    sum_mat = vector1 - vector2
    return np.sqrt(np.power(sum_mat[0, 0], 2) + np.power(sum_mat[0, 1], 2) + np.power(sum_mat[0, 2], 2))


def orientationMean(vector_mat):
    std_vec = vector_mat[0, :]
    sum_vec = np.zeros((1, 3))
    m = vector_mat.shape[0]
    for i in range(m):
        cos_vector = np.multiply(std_vec, vector_mat[i, :])
        sum_cos = np.sum(cos_vector)
        if sum_cos < 0:
            vector_mat[i, 0] = np.multiply(vector_mat[i, 0], -1)
            vector_mat[i, 1] = np.multiply(vector_mat[i, 1], -1)
            vector_mat[i, 2] = np.multiply(vector_mat[i, 2], -1)
            # I don't know why, but the sentence below make the result worse
            # vector_mat[i, :] = np.multiply(vector_mat[i, :], -1)
        sum_vec = sum_vec + vector_mat[i, :]
    mean_vec = np.mean(vector_mat, axis=0)
    return mean_vec
