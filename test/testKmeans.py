from importlib import reload
import numpy as np
import plotGraph as plt, helper as hp
import kMeans as km

# ####### Parameters #########
k = 3  # number of cluster
max_iterate = 50  # maximum round of iterate to stop
min_error = 0  # minimum difference between two iterate's error value to stop
calc_round = 1  # round of running k-means Algorithm to get the best result
file_name = "_dataSet.txt"  # file name of Test Data, should be in the same folder of program
plot_origin = 0  # plot origin graph or clustered graph, 1 = origin, else = clustered
rand_method = 0  # method of create initial centroids, 0 = remotest centroids, 1 = random centroids
sieve_angle = 90  # Data which intersection angle with mean vector larger than that will be removed
show_ini_centroids = False  # whether to show initial random centroids
# ############################

reload(km)
reload(plt)
reload(hp)

test_data = np.mat(hp.loadDataSet(file_name))

if plot_origin == 1:
    test_data_show = hp.degree2radian(test_data, 0)
    plt.createOrigin(test_data_show.A)
else:
    # get and transfer Data
    test_data_radian = hp.degree2radian(test_data, -1)
    test_data_vector = hp.orientation2vector(test_data_radian)

    # run k-means Algorithm
    best_ini_centroids = km.randCenter(test_data_vector, k)
    best_centroids = km.randCenter(test_data_vector, k)
    best_cluster = np.zeros((test_data_vector.shape[0], 2))
    min_mean_error = np.inf
    print("")
    for i in range(calc_round):
        print("************ round ", i + 1, " of calculating ************")
        print("")
        if rand_method == 0:
            method = km.remotestCenter
        else:
            method = km.randCenter
        ini_centroids, my_centroids, cluster_result = km.angKmeans(test_data_vector, k,
                                                                   create_cent=method,
                                                                   calc_mean=hp.orientationMean,
                                                                   max_iterate=max_iterate,
                                                                   min_error=min_error)
        mean_error = np.sum(cluster_result[:, 1], axis=0) / cluster_result.shape[0]
        print("")
        print("average error of this round: ", mean_error)
        print("centroid(vector) of this round:")
        print(my_centroids)
        print("")
        if mean_error < min_mean_error:
            min_mean_error = mean_error
            best_ini_centroids = ini_centroids
            best_centroids = my_centroids
            best_cluster = cluster_result
    ini_centroids = best_ini_centroids
    my_centroids = best_centroids
    cluster_result = best_cluster
    print("*************** calculating finished ***************")
    print("")
    print("minimum average error: ", min_mean_error)
    print("")

    # transfer and sieve result
    result_data = np.c_[test_data, cluster_result]
    result_data = hp.degree2radian(result_data, 0)
    my_centroids_orientation = hp.vector2orientation(my_centroids)
    my_centroids_orientation_radian = hp.degree2radian(my_centroids_orientation, 0)
    sieve_vector = hp.sieveDataIndex(test_data_vector, my_centroids, sieve_angle)
    result_data_sieve = hp.sieveDataByIndex(result_data, sieve_vector)

    ini_centroids_orientation = hp.vector2orientation(ini_centroids)
    ini_centroids_orientation_radian = hp.degree2radian(ini_centroids_orientation, 0)

    # print ang plot result
    print("****************** cluster result ******************")
    print("")
    print("matrix of [direction(radian) angle clusterIndex error]:")
    print(result_data_sieve)
    print("")
    print("********************* centroids ********************")
    print("")
    print("matrix of [direction(degree) angle]:")
    print(my_centroids_orientation)
    print("")
    if show_ini_centroids:
        plt.createClusterKmeansWith2Centroids(result_data_sieve.A, my_centroids_orientation_radian.A,
                                              ini_centroids_orientation_radian.A)
    else:
        plt.createClusterKmeansWith1Centroids(result_data_sieve.A, my_centroids_orientation_radian.A)
