from importlib import reload
import numpy as np
import plotGraph as plt, helper as hp
import dbscan as dbs

# ####### Parameters #########
eps = 0.05
min_pts = 10
file_name = "_dataSet.txt"  # file name of Test Data, should be in the same folder of program
plot_origin = 0  # plot origin graph or clustered graph, 1 = origin, else = clustered
sieve_angle = 90  # Data which intersection angle with mean vector larger than that will be removed
# ############################

reload(dbs)
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

    # run dbscan Algorithm
    print("******************* start dbscan ********************")
    print("")
    cluster_result, noise_result, k = dbs.dbscan(test_data_vector, eps, min_pts)

    # transfer result
    result_data = np.mat(np.zeros((cluster_result.shape[0], 3)))
    result_data[:, 0:2] = hp.vector2orientation(cluster_result[:, 0:3])
    result_data[:, -1] = cluster_result[:, -1]
    result_data = hp.degree2radian(result_data, 0)
    noise_data = hp.vector2orientation(noise_result[:, 0:3])
    noise_data = hp.degree2radian(noise_data, 0)

    # print ang plot result
    print("****************** cluster result ******************")
    print("")
    print(result_data)
    print("")
    print("****************** cluster result ******************")
    print("")
    print(noise_data)
    print("number of cluster: ", k)
    plt.createClusterDBSCAN(result_data.A, noise_data.A)
