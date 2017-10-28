import numpy as np
import kMeans


def ini_param(data_set, k, cov_value=0.5):
    alpha = 1 / k
    mean = kMeans.remotestCenter(data_set, k)
    cov = np.mat(np.zeros((data_set.shape[1], data_set.shape[1])))
    for i in range(data_set.shape[1]):
        cov[i, i] = cov_value
    return alpha, mean, cov


def pdf_gaussion(x, mean, cov):
    m = 1
    cos_vector = np.multiply(x, mean)
    sum_cos = np.sum(cos_vector)
    if sum_cos < 0:
        m = -1
    temp = m * x - mean
    t = temp.T
    cr = cov.I
    m = (-0.5) * (temp * cr * t)
    top = np.power(np.e, m)
    n = x.shape[0]
    d = np.linalg.det(cov.A)
    d = np.power(d, 0.5)
    bot = np.power(2 * np.pi, n / 2) * d
    return top / bot


def gmm(data_set, k, max_iter):
    a, m, c = ini_param(data_set, k)
    alpha = []
    mean = m
    cov = []
    for i in range(k):
        alpha.append(a)
        cov.append(c)
    count_iter = 0
    p_ji = np.mat(np.zeros((data_set.shape[0], k)))
    while count_iter <= max_iter:
        for j in range(data_set.shape[0]):
            p_temp = []
            for i in range(k):
                temp = alpha[i] * pdf_gaussion(data_set[j, :], mean[i, :], cov[i])
                p_temp.append(temp)
            p_sum = 0
            for i in range(len(p_temp)):
                p_sum = p_sum + p_temp[i]
            for i in range(k):
                p_ji[j, i] = p_temp[i] / p_sum
        sum_pj = np.sum(p_ji, axis=0)
        for i in range(k):
            # compute new mean vector
            sum_pjx = np.mat(np.zeros((1, data_set.shape[1])))
            for jj in range(data_set.shape[0]):
                sum_pjx = sum_pjx + p_ji[jj, i] * data_set[jj, :]
            mean[i, :] = sum_pjx / sum_pj[0, i]
            # compute new cov matrix
            sum_covx = np.mat(np.zeros((data_set.shape[1], data_set.shape[1])))
            for jj in range(data_set.shape[0]):
                m = 1
                cos_vector = np.multiply(data_set[jj, :], mean[i, :])
                sum_cos = np.sum(cos_vector)
                if sum_cos < 0:
                    m = -1
                x_u = m * data_set[jj, :] - mean[i, :]
                x_u_t = x_u.T
                sum_covx = sum_covx + p_ji[jj, i] * x_u_t * x_u
            cov[i] = sum_covx / sum_pj[0, i]
            # compute new alpha value
            alpha[i] = sum_pj[0, i] / data_set.shape[0]
        count_iter = count_iter + 1
        print("iter ", count_iter, " complete")
    cluster_label = []
    for j in range(data_set.shape[0]):
        max_val = p_ji[j, 0]
        max_idx = 0
        for i in range(k):
            if p_ji[j, i] > max_val:
                max_val = p_ji[j, i]
                max_idx = i
        cluster_label.append(max_idx)
    return mean, cluster_label
