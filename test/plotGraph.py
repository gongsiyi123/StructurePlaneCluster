import matplotlib.pyplot as plt
import numpy as np


def createOrigin(data_set):
    ax1 = plt.subplot(111, projection='polar')
    ax1.set_theta_direction(-1)
    ax1.set_theta_zero_location('N')
    ax1.set_thetagrids(np.arange(0.0, 360.0, 30.0))
    ax1.set_rlim(0, 90)
    ax1.set_rlabel_position('45')
    ax1.scatter(data_set[:, 0], data_set[:, 1], c='k', marker='o', s=5)
    plt.show()


def createClusterKmeansWith1Centroids(data_set, centroids):
    new_set = data_set.copy()
    ax = plt.subplot(111, projection='polar')
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    ax.set_thetagrids(np.arange(0.0, 360.0, 30.0))
    ax.set_rlim(0, 90)
    ax.set_rlabel_position('45')
    ax.scatter(new_set[:, 0], new_set[:, 1], c='k', marker='o', s=5)
    color = np.random.randn(new_set.shape[0])
    color = color.astype(np.str)
    dic_color = {}
    range_color = 100
    for i in range(new_set.shape[0]):
        if new_set[i, 2] not in dic_color.keys():
            rand_color = np.random.randint(0, range_color)
            while rand_color in dic_color.values():
                rand_color = np.random.randint(0, range_color)
            dic_color[new_set[i, 2]] = rand_color
    for i in range(new_set.shape[0]):
        color[i] = dic_color[new_set[i, 2]]
    ax.scatter(new_set[:, 0], new_set[:, 1], c=color, marker='^', s=25)
    ax.scatter(centroids[:, 0], centroids[:, 1], c='r', marker='x', s=200)
    plt.show()


def createClusterKmeansWith2Centroids(data_set, centroids, ini_centroids):
    new_set = data_set.copy()
    ax = plt.subplot(111, projection='polar')
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    ax.set_thetagrids(np.arange(0.0, 360.0, 30.0))
    ax.set_rlim(0, 90)
    ax.set_rlabel_position('45')
    ax.scatter(new_set[:, 0], new_set[:, 1], c='k', marker='o', s=5)
    color = np.random.randn(new_set.shape[0])
    color = color.astype(np.str)
    dic_color = {}
    range_color = 16
    for i in range(new_set.shape[0]):
        if new_set[i, 2] not in dic_color.keys():
            rand_color = np.random.randint(0, range_color)
            while rand_color in dic_color.values():
                rand_color = np.random.randint(0, range_color)
            dic_color[new_set[i, 2]] = rand_color
    for i in range(new_set.shape[0]):
        color[i] = dic_color[new_set[i, 2]]
    ax.scatter(new_set[:, 0], new_set[:, 1], c=color, marker='^', s=25)
    ax.scatter(centroids[:, 0], centroids[:, 1], c='r', marker='x', s=200)
    ax.scatter(ini_centroids[:, 0], ini_centroids[:, 1], c='b', marker='+', s=200)
    plt.show()


def createClusterDBSCAN(cluster_set, noise_set):
    new_set = cluster_set.copy()
    ax = plt.subplot(111, projection='polar')
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    ax.set_thetagrids(np.arange(0.0, 360.0, 30.0))
    ax.set_rlim(0, 90)
    ax.set_rlabel_position('45')
    color = np.random.randn(new_set.shape[0])
    color = color.astype(np.str)
    dic_color = {}
    range_color = 100
    for i in range(new_set.shape[0]):
        if new_set[i, 2] not in dic_color.keys():
            rand_color = np.random.randint(0, range_color)
            while rand_color in dic_color.values():
                rand_color = np.random.randint(0, range_color)
            dic_color[new_set[i, 2]] = rand_color
    for i in range(new_set.shape[0]):
        color[i] = dic_color[new_set[i, 2]]
    ax.scatter(new_set[:, 0], new_set[:, 1], c=color, marker='^', s=25)
    ax.scatter(noise_set[:, 0], noise_set[:, 1], c='k', marker='x', s=50)
    plt.show()


def createClusterAfinityPropagation(data_set, centroids, clus_list):
    ax = plt.subplot(111, projection='polar')
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    ax.set_thetagrids(np.arange(0.0, 360.0, 30.0))
    ax.set_rlim(0, 90)
    ax.set_rlabel_position('45')
    color = np.random.randn(len(clus_list))
    color = color.astype(np.str)
    dic_color = {}
    range_color = 100
    for i in range(len(clus_list)):
        if clus_list[i] not in dic_color.keys():
            rand_color = np.random.randint(0, range_color)
            while rand_color in dic_color.values():
                rand_color = np.random.randint(0, range_color)
            dic_color[clus_list[i]] = rand_color
    for i in range(len(clus_list)):
        color[i] = dic_color[clus_list[i]]
    ax.scatter(data_set[:, 0].A, data_set[:, 1].A, c=color, marker='^', s=25)
    ax.scatter(centroids[:, 0].A, centroids[:, 1].A, c='r', marker='x', s=200)
    plt.show()
