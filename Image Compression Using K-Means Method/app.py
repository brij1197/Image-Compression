# Creating an image compression program using K-Means(Unsupervised Learning basically trial and error)

import os
import sys

from PIL import Image
import numpy as np

# Creating Initial points for centroids


def initialize_k_centroids(X, K):
    # Choose K Points from X at random
    m = len(X)
    return X[np.random.choice(m, K, replace=False), :]

# Defining the function to load image taken as parameter


def load_image(path):
    """ Load image from path. Return a numpy array """
    image = Image.open(path)
    return np.asarray(image) / 255
# Function to find the centroid for each training example


def find_closest_centroids(X, centroids):
    m = len(X)
    c = np.zeros(m)
    for i in range(m):
        # Find distances
        distances = np.linalg.norm(X[i] - centroids, axis=1)

        # Assign closest cluster to c[i]
        c[i] = np.argmin(distances)

    return c

# Compute the distance of each example to its centroid and take the average of distance for every centroid


def compute_means(X, idx, K):
    _, n = X.shape
    centroids = np.zeros((K, n))
    for k in range(K):
        examples = X[np.where(idx == k)]
        mean = [np.mean(column) for column in examples.T]
        centroids[k] = mean
    return centroids

# Set max iterations to 10. If centroids aren't moving anymore, we return the results as we cannot optimize any further


def find_k_means(X, K, max_iters=10):
    centroids = initialize_k_centroids(X, K)
    previous_centroids = centroids
    currentiteration = 0
    for _ in range(max_iters):
        currentiteration += 1
        print("Progress : ", (currentiteration/max_iters)*100,"%")
        idx = find_closest_centroids(X, centroids)
        centroids = compute_means(X, idx, K)
        if(centroids == previous_centroids).all():
            # The centroids are not moving anymore
            return centroids
        else:
            previous_centroids = centroids
    return centroids, idx


def main():
    try:
        image_path = sys.argv[1]
        assert os.path.isfile(image_path)
    except (IndexError, AssertionError):
        print("Please specify an image")

    image = load_image(image_path)
    w, h, d = image.shape
    print("Image found with width: {}, height: {}, depth: {}".format(w, h, d))

    # We are reshaping the image because each pixel has the same meaning (color), so they don’t have to be presented as a grid.
    X = image.reshape((w*h, d))
    K = 20  # Desired number of colors in compressed image

    # Colors are chosen by an algorithm
    colors, _ = find_k_means(X, K, max_iters=10)
    # Compute the inddexdes of the current colors
    print("Constructing compressed image")
    idx = find_closest_centroids(X, colors)

    # Once we have the data required, we reconstruct the image by substituting the color index with the color and reshaping the image back to its orig. dimensions
    idx = np.array(idx, dtype=np.uint8)
    X_reconstructed = np.array(colors[idx, :] * 255, dtype=np.uint8).reshape((w, h, d))
    compressed_image = Image.fromarray(X_reconstructed)
    compressed_image.save('compressed.jpg')
    print("Image Created")


if __name__ == "__main__":
    main()
