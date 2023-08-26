@add_metaclass(ABCMeta)
class PartitionalClusteringObjectiveFunction(ObjectiveFunction):

    def __init__(self, dim, n_clusters, data):
        super(PartitionalClusteringObjectiveFunction, self)\
            .__init__('PartitionalClusteringObjectiveFunction', dim, 0.0, 1.0)
        self.n_clusters = n_clusters
        self.centroids = {}
        self.data = data

    def decode(self, x):
        centroids = x.reshape(self.n_clusters, self.dim)
        self.centroids = dict(enumerate(centroids))

    @abstractmethod
    def evaluate(self, x):
        pass
     
class SumOfSquaredErrors(PartitionalClusteringObjectiveFunction):

    def __init__(self, dim, n_clusters, data):
        super(SumOfSquaredErrors, self).__init__(dim, n_clusters, data)
        self.name = 'SumOfSquaredErrors'

    def evaluate(self, x):
        self.decode(x)

        clusters = {key: [] for key in self.centroids.keys()}
        for instance in self.data:
            distances = [np.linalg.norm(self.centroids[idx] - instance)
                         for idx in self.centroids]
            clusters[np.argmin(distances)].append(instance)

        sum_of_squared_errors = 0.0
        for idx in self.centroids:
            distances = [np.linalg.norm(self.centroids[idx] - instance)
                         for instance in clusters[idx]]
            sum_of_squared_errors += sum(np.power(distances, 2))
        return sum_of_squared_errors