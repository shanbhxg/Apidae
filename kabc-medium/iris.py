import matplotlib.pyplot as plt

from artibc import ABC
import numpy as np
from objective_function import SumOfSquaredErrors

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

data = MinMaxScaler().fit_transform(load_iris()['data'][:, [1,3]])
plt.figure(figsize=(9,8))
plt.scatter(data[:,0], data[:,1], s=50, edgecolor='w', alpha=0.5)
plt.title('Original Data')

colors = ['r', 'g', 'y']
target = load_iris()['target']

plt.figure(figsize=(9,8))
for instance, tgt in zip(data, target):
    plt.scatter(instance[0], instance[1], s=50,
                edgecolor='w', alpha=0.5, color=colors[tgt])
plt.title('Original Groups')


objective_function = SumOfSquaredErrors(dim=6, n_clusters=3, data=data)
optimizer = ABC(obj_function=objective_function, colony_size=30,
                n_iter=300, max_trials=100)
optimizer.optimize()

def decode_centroids(centroids, n_clusters, data):
    return centroids.reshape(n_clusters, data.shape[1])
  
centroids = dict(enumerate(decode_centroids(optimizer.optimal_solution.pos,
                                            n_clusters=3, data=data)))

def assign_centroid(centroids, point):
    distances = [np.linalg.norm(point - centroids[idx]) for idx in centroids]
    return np.argmin(distances)
  
custom_tgt = []
for instance in data:
    custom_tgt.append(assign_centroid(centroids, instance))

colors = ['r', 'g', 'y']
plt.figure(figsize=(9,8))
for instance, tgt in zip(data, custom_tgt):
    plt.scatter(instance[0], instance[1], s=50, edgecolor='w',
                alpha=0.5, color=colors[tgt])

for centroid in centroids:
    plt.scatter(centroids[centroid][0], centroids[centroid][1],
                color='k', marker='x', lw=5, s=500)
plt.title('Partitioned Data found by ABC')

itr = range(len(optimizer.optimality_tracking))
val = optimizer.optimality_tracking
plt.figure(figsize=(10, 9))
plt.plot(itr, val)
plt.title('Sum of Squared Errors')
plt.ylabel('Fitness')
plt.xlabel('Iteration')

