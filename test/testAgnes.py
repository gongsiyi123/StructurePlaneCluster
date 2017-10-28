from importlib import reload
import numpy as np
import plotGraph as plt, helper as hp
import agnes as ag

# ####### Parameters #########
k = 3  # number of cluster
file_name = "_dataSet.txt"  # file name of Test Data, should be in the same folder of program
plot_origin = 0  # plot origin graph or clustered graph, 1 = origin, else = clustered
sieve_angle = 35  # Data which intersection angle with mean vector larger than that will be removed
# ############################

reload(ag)
reload(plt)
reload(hp)

test_data = np.mat(hp.loadDataSet(file_name))

if plot_origin == 1:
    test_data_show = hp.degree2radian(test_data, 0)
    plt.createOrigin(test_data_show.A)
else:
    # get and transfer data
    test_data_radian = hp.degree2radian(test_data, -1)
    test_data_vector = hp.orientation2vector(test_data_radian)

    # run gnes algorithm
    print("*************** calculating started ***************")
    print("")
    cluster_centroids, cluster_list = ag.agnes(test_data_vector, k, dist_calc=ag.dist_min)

    print("")
    print("*************** calculating finished ***************")
    print("")
    print("cluster centroids: ")
    print(cluster_centroids)
    print("")
    print("cluster list: ")
    print(cluster_list)
    print("")

    # transfer and sieve result
    test_data_radian = hp.degree2radian(test_data, 0)
    my_centroids_orientation = hp.vector2orientation(cluster_centroids)
    my_centroids_orientation_radian = hp.degree2radian(my_centroids_orientation, 0)

    # print ang plot result
    print("********************* centroids ********************")
    print("")
    print("matrix of [direction(degree) angle]:")
    print(my_centroids_orientation)
    print("")
    plt.createClusterAfinityPropagation(test_data_radian, my_centroids_orientation_radian, cluster_list)
