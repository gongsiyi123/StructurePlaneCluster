from importlib import reload
import numpy as np
import plotGraph as plt, helper as hp
import affinityPropagation as ap

# ####### Parameters #########
file_name = "_dataSet.txt"  # file name of Test Data, should be in the same folder of program
damping = 0.8  # damping parameter of affinity propagation, should between 0 to 1
max_iterate = 100  # maximum round of iterate to stop
plot_origin = 0  # plot origin graph or clustered graph, 1 = origin, else = clustered
# ############################

reload(ap)
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

    # run affinity propagation Algorithm
    print("*************** calculating started ***************")
    print("")
    cluster_centroids, cluster_list = ap.affinityPropagation(test_data_vector, damping, max_iterate)
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
