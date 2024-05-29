import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean


# Function to pad arrays to the specified length
def pad_array(arr, length):
    if len(arr) > length:
        return arr[:length]
    return np.pad(arr, (0, length - len(arr)), 'constant')


def clustering_results(results, max_generations):
    # Assuming each object is a tuple of three arrays: (array1, array2, array3)
    objects = results

    # Pad each array in each tuple
    padded_objects = [
        (pad_array(obj[0], max_generations), pad_array(obj[1], max_generations), pad_array(obj[2], max_generations))
        for obj in objects
    ]

    # Flatten each object to be a single vector
    flattened_objects = [np.concatenate(obj) for obj in padded_objects]

    # Convert list to numpy array
    X = np.array(flattened_objects)

    # Define the number of clusters (this is a parameter you might need to tune)
    num_clusters = 5

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)

    # Assign objects to clusters
    labels = kmeans.labels_

    cluster_averages = []

    # Create an average object to represent each cluster (consists of 3 tuples)
    for cluster in range(num_clusters):
        selected_indices = np.where(labels == cluster)[0]
        array1, array2, array3 = [], [], []
        for i in range(max_generations):
            val1 = sum(padded_objects[j][0][i] for j in selected_indices) / selected_indices.__len__()
            val2 = sum(padded_objects[j][1][i] for j in selected_indices) / selected_indices.__len__()
            val3 = sum(padded_objects[j][2][i] for j in selected_indices) / selected_indices.__len__()
            array1.append(val1)
            array2.append(val2)
            array3.append(val3)
        cluster_averages.append([array1, array2, array3])

    # cluster_averages now contains the average arrays for each cluster
    return cluster_averages
