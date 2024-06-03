
import math
import numpy as np
from pyspark.sql.functions import col, split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def extract_times_to_seconds(rdd):
    """
    Extracts the start and end times of day from the 'started_at' and 'ended_at' columns
    and converts them to seconds.
    """
    # Split started_at and ended_at columns into date and time
    rdd = rdd.withColumn("start_date", split(col("started_at"), " ")[0]) \
          .withColumn("start_time", split(col("started_at"), " ")[1]) \
          .withColumn("end_date", split(col("ended_at"), " ")[0]) \
          .withColumn("end_time", split(col("ended_at"), " ")[1])


    # Drop the original started_at and ended_at columns
    rdd = rdd.drop("started_at", "ended_at")


    # Split the time strings into hours, minutes, and seconds
    rdd = rdd.withColumn("start_time_split", split(col("start_time"), ":")) \
          .withColumn("end_time_split", split(col("end_time"), ":"))


    # Convert hours, minutes, seconds to seconds
    rdd = rdd.withColumn("start_seconds",
                      col("start_time_split")[0].cast("int") * 3600 +
                      col("start_time_split")[1].cast("int") * 60 +
                      col("start_time_split")[2].cast("int")) \
          .withColumn("end_seconds",
                      col("end_time_split")[0].cast("int") * 3600 +
                      col("end_time_split")[1].cast("int") * 60 +
                      col("end_time_split")[2].cast("int"))


    # Drop intermediate columns
    rdd = rdd.drop("start_time_split", "end_time_split")


    # Drop all date and time columns EXCEPT for the new 'seconds' columns
    rdd = rdd.drop("start_date", "start_time", "end_date", "end_time")


    return rdd

# Function to compute the true distance between two points on Earth's surface

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the Earth (specified in decimal degrees)
    """
    # Convert latitude and longitude from degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)


    # Radius of the Earth in kilometers
    radius = 6371.0


    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = radius * c

    return distance

def starts_dist(u, v):
    """
    Compute the distance between the starting points of two trips.
    """

    lat1 = u[0]
    lon1 = u[1]
    lat2 = v[0]
    lon2 = v[1]

    return haversine(lat1, lon1, lat2, lon2)


def ends_dist(u, v):
    """
    Compute the distance between the ending points of two trips.
    """
    lat1 = u[2]
    lon1 = u[3]
    lat2 = v[2]
    lon2 = v[3]

    return haversine(lat1, lon1, lat2, lon2)


def sine_vecs(a, b):
    """
    Compute the sine of the angle between two vectors.
    """
    return np.abs(np.cross(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))

def calculate_initial_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate the initial bearing between two points on Earth.
    """
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlon = lon2 - lon1

    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    y = math.sin(dlon) * math.cos(lat2)

    initial_bearing = math.atan2(y, x)

    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing

def sine_bearings(u, v):
    """
    Compute the sine of the difference between the bearings of two trips.
    """
    u_bearing = calculate_initial_bearing(u[0], u[1], u[2], u[3])
    v_bearing = calculate_initial_bearing(v[0], v[1], v[2], v[3])

    # Convert to radians
    u_bearing = math.radians(u_bearing)
    v_bearing = math.radians(v_bearing)

    return np.sin(u_bearing - v_bearing)


def spatial_cost(u, v, gamma):
    """
    Compute the spatial cost between two trips.
    """
    # Calculate the cost according to the formula
    cost = starts_dist(u, v) + ends_dist(u, v) + gamma * np.abs(sine_bearings(u, v))

    return cost

def K_means_time_and_space(data, centroids, max_iter, norm, lam, gam):
    """
    Perform K-means clustering with a custom distance metric that incorporates
    both time and spatial information.
    """
    # The number of seconds in a day (will need later):
    s = 24 * 60 * 60

    # set the number of iterations for K-means
    MAX_ITER = max_iter + 1

    # initialize the cost associated with each iteration
    costs = np.zeros(MAX_ITER)

    # outer-most loop for k-means iterations
    for t in np.arange(MAX_ITER):

        # will store the index corresponding to the cluster into which each
        # point is partitioned
        clusters = np.zeros(data.shape[0])

        # will store the new centroids computed from each iteration
        new_centroids = np.zeros([centroids.shape[0], centroids.shape[1]])

        # will store the count of the number of points falling into each
        # cluster
        point_count = np.zeros([centroids.shape[0], 1])


        # loop over each row in the data
        for j in np.arange(data.shape[0]):

            # will store the custom distance from the current row of the data
            # to each centroid
            dists = np.zeros(centroids.shape[0])


            # loop over each centroid
            for i in np.arange(centroids.shape[0]):

              # get the current row, centroid pair
              x_row = data[j]
              c_row = centroids[i]

              # start by calculating the time cost
              x_start_time = x_row[4]
              c_start_time = c_row[4]

              # perform the wrap-around, finding the minimum distance in 24-hour
              # time
              dists[i] = lam * np.min([np.abs(x_start_time - c_start_time),
                                 s - np.abs(x_start_time - c_start_time)])


              # now add the spatial cost
              dists[i] += (1 - lam) * spatial_cost(x_row[:4], c_row[:4], gam)


            # determine the (index of the) cluster with the closest centroid
            # to x_row
            clusters[j] = np.argmin(dists)

            # add the current x_row to the row in the array of new centroids
            # corresponding to the cluster x_row falls into
            new_centroids[int(clusters[j]), :] += x_row
            point_count[int(clusters[j])] += 1

            # store the cost(s) associated with the current x_row
            costs[t] += np.power(dists[int(clusters[j])], norm)

        # divide by the number of points in each cluster to obtain the new
        # centroids
        new_centroids = new_centroids / point_count
        centroids = new_centroids

    # return the costs, new centroids, and cluster assignments
    return costs, new_centroids, clusters

def K_means_time_and_space_alt(data, centroids, max_iter, norm, lam, gam, dist_func):

    # The number of seconds in a day (will need later):
    s = 24 * 60 * 60

    # set the number of iterations for K-means
    MAX_ITER = max_iter + 1

    # initialize the cost associated with each iteration
    costs = np.zeros(MAX_ITER)

    # outer-most loop for k-means iterations
    for t in np.arange(MAX_ITER):

        # will store the index corresponding to the cluster into which each
        # point is partitioned
        clusters = np.zeros(data.shape[0])

        # will store the new centroids computed from each iteration
        new_centroids = np.zeros([centroids.shape[0], centroids.shape[1]])

        # will store the count of the number of points falling into each
        # cluster
        point_count = np.zeros([centroids.shape[0], 1])


        # loop over each row in the data
        for j in np.arange(data.shape[0]):

            # will store the custom distance from the current row of the data
            # to each centroid
            dists = np.zeros(centroids.shape[0])


            # loop over each centroid
            for i in np.arange(centroids.shape[0]):

              # get the current row, centroid pair
              x_row = data[j]
              c_row = centroids[i]

              # calculate distance
              dists[i] = dist_func(x_row[:5], c_row[:5], lam, gam)

            # determine the (index of the) cluster with the closest centroid
            # to x_row
            clusters[j] = np.argmin(dists)

            # add the current x_row to the row in the array of new centroids
            # corresponding to the cluster x_row falls into
            new_centroids[int(clusters[j]), :] += x_row
            point_count[int(clusters[j])] += 1

            # store the cost(s) associated with the current x_row
            costs[t] += np.power(dists[int(clusters[j])], norm)

        # divide by the number of points in each cluster to obtain the new
        # centroids
        new_centroids = new_centroids / point_count
        centroids = new_centroids

    # return the costs, new centroids, and cluster assignments
    return costs, new_centroids, clusters

def custom_distance_metric(x_row, c_row, lam, gam):
    """
    Compute the custom distance between a data point and a centroid.
    """
    dist = 0
    s = 24 * 60 * 60

    # start by calculating the time cost
    x_start_time = x_row[4]
    c_start_time = c_row[4]

    # perform the wrap-around, finding the minimum distance in 24-hour
    # time
    dist = lam * np.min([np.abs(x_start_time - c_start_time),
                                 s - np.abs(x_start_time - c_start_time)])

    # now add the spatial cost
    dist += (1 - lam) * spatial_cost(x_row[:4], c_row[:4], gam)

    return dist

def euclidean_dist(xrow, crow, lam=None, gam=None):
    return np.linalg.norm(xrow-crow)

def silhouette_score_custom(X, labels, custom_distance, lam, gam):
    """
    Compute the silhouette score for a clustering with a custom distance metric.
    """
    n_samples = len(X)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Initialize silhouette scores
    silhouette_scores = np.zeros(n_samples)

    # Compute a and b for each sample
    for i in range(n_samples):
        # Points in the same cluster
        same_cluster = (labels == labels[i])
        same_cluster[i] = False  # exclude the point itself

        # Compute a (mean intra-cluster distance)
        a = np.mean([custom_distance(X[i], X[j], lam, gam) for j in range(n_samples) if same_cluster[j]])

        # Compute b (mean nearest-cluster distance)
        b = np.inf
        for label in unique_labels:
            if label == labels[i]:
                continue
            other_cluster = (labels == label)
            mean_distance = np.mean([custom_distance(X[i], X[j], lam, gam) for j in range(n_samples) if other_cluster[j]])
            if mean_distance < b:
                b = mean_distance

        # Silhouette score for the sample
        silhouette_scores[i] = (b - a) / np.max([a, b])

    # Overall silhouette score
    overall_silhouette_score = np.mean(silhouette_scores)
    return overall_silhouette_score, silhouette_scores

def cluster_train_test_split(small_time_and_space_arr_pd, cluster_col = "cluster_custom", test_size=0.2, random_state=42):
    """
    Splits the data into train and test sets for each cluster.
    """
    # Get the unique clusters
    unique_clusters = np.unique(small_time_and_space_arr_pd[cluster_col])

    # Dictionary to store train/test splits for each cluster
    train_test_splits = {}
    scalers = {}

    for cluster in unique_clusters:
        # Filter rows with the current cluster value
        cluster_data = small_time_and_space_arr_pd[small_time_and_space_arr_pd[cluster_col] == cluster]

        X = cluster_data.drop(columns=[cluster_col, "end_time"])
        y = cluster_data["end_time"]
    
        # Split the data into train and test sets
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Normalize the output data
        scaler_input = StandardScaler()
        scaler_input.fit(train_x)
        train_x = scaler_input.transform(train_x)
        test_x = scaler_input.transform(test_x)

        scaler_output = StandardScaler()
        scaler_output.fit(train_y.values.reshape(-1, 1))
        train_y = scaler_output.transform(train_y.values.reshape(-1, 1)).flatten()
        test_y = scaler_output.transform(test_y.values.reshape(-1, 1)).flatten()

        # Store the train/test splits for the current cluster
        train_test_splits[cluster] = (train_x, test_x, train_y, test_y)

        # Store the scalers for the current cluster
        scalers[cluster] = (scaler_input, scaler_output)

        # In this, train_test_splits[0] will contain the splits for cluster 0
        # train_test_splits[0][0] will contain the input training data for cluster 0
        # train_test_splits[0][1] will contain the input testing data for cluster 0
        # train_test_splits[0][2] will contain the output training data for cluster 0
        # train_test_splits[0][3] will contain the output testing data for cluster 0

    return train_test_splits, scalers

