import os
import dataset
import numpy
from functools import partial
from scipy.optimize import minimize
import scipy
from sklearn.cluster import KMeans
import tensorflow as tf

def radial_basis_phi(x, y, b, v):
    norm = numpy.linalg.norm(x - y)
    return max(1.0 - norm / b, 0) ** v


def squared_exponential(x, y, sig=0.8, sig2=1):
    norm = numpy.linalg.norm(x - y)
    dist = norm * norm
    return numpy.exp(- dist / (2 * sig * sig2))


def compute_affinity(X, kernel=squared_exponential):
    N = X.shape[0]
    ans = numpy.zeros((N, N))
    for i in range(N):
        for j in range(N):
            ans[i][j] = kernel(X[i], X[j])
    return ans


def com_aff_local_scaling(X):
    N = X.shape[0]
    ans = numpy.zeros((N, N))
    sig = []
    for i in range(N):
        dists = []
        for j in range(N):
            dists.append(numpy.linalg.norm(X[i] - X[j]))
        dists.sort()
        sig.append(numpy.mean(dists[:5]))

    for i in range(N):
        for j in range(N):
            ans[i][j] = squared_exponential(X[i], X[j], sig[i], sig[j])
    return ans


def _log_sq(x, eps=1e-14):
    t = numpy.log(x + eps)
    return t * t


def _auto_prunning_cost(X, K, b, v, gamma=0.5):
    kernel = partial(radial_basis_phi, b=b, v=v)
    K_bv = compute_affinity(X, kernel)
    num = numpy.linalg.norm(K_bv - K)
    den = numpy.sqrt(numpy.linalg.norm(K_bv) * numpy.linalg.norm(K))
    rho = _log_sq(num / den)
    n = X.shape[0]
    s = 1.0 - (numpy.count_nonzero(K_bv) / (n * n))
    s = _log_sq(s)
    return numpy.sqrt((1.0 - gamma) * rho + gamma * s)


def _auto_prunning_find_b(X, v, affinity):
    K = affinity(X)

    def cost_b(x):
        return -1 * _auto_prunning_cost(X, K, x, v)  # we need to maximize this function

    result = minimize(
        cost_b,
        [numpy.mean(K)],
        bounds=((0, None),),  # positive
        # options={'disp': True, 'maxiter': 100})
    )
    print('best b ', result.x[0])
    return result.x[0]


def automatic_prunning(X, affinity=com_aff_local_scaling):
    D = X.shape[1]
    v = (D + 1) / 2
    b = _auto_prunning_find_b(X, v, affinity)

    affinity = affinity(X)
    N = X.shape[0]
    ans = numpy.zeros((N, N))
    for i in range(N):
        for j in range(N):
            ans[i][j] = radial_basis_phi(X[i], X[j], b, v) * affinity[i][j]
    return ans


def laplacian(A):
    """Computes the symetric normalized laplacian.
    L = D^{-1/2} A D{-1/2}
    """
    D = numpy.zeros(A.shape)
    w = numpy.sum(A, axis=0)
    D.flat[::len(w) + 1] = w ** (-0.5)  # set the diag of D to w
    return D.dot(A).dot(D)


def k_means(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=1231)
    return kmeans.fit(X).labels_


def spectral_clustering(affinity, n_clusters, cluster_method=k_means):
    L = laplacian(affinity)
    eig_val, eig_vect = scipy.sparse.linalg.eigs(L, n_clusters)
    X = eig_vect.real
    rows_norm = numpy.linalg.norm(X, axis=1, ord=2)
    Y = (X.T / rows_norm).T
    labels = cluster_method(Y, n_clusters)
    return labels


# class info
classes = sorted(os.listdir('data/256_ObjectCategories'))
classes.remove('257.clutter')

# Number of color channels for the images: 1 channel for gray-scale.
num_channels = 3
# image dimensions (only squares for now)
img_size = 128
# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels
# validation split
validation_size = 0.7

train_path = 'data/256_ObjectCategories/'
test_path = 'data/256_ObjectCategories/257.clutter/'
checkpoint_dir = "models/"

# ## Load Data
data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)
# print(data.train.labels.shape)
sess = tf.Session()
lables = tf.argmax(data.train.labels, dimension=1).eval(session=sess)
print(lables.shape)
print(lables)
print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Validation-set:\t{}".format(len(data.valid.labels)))

X = data.train.images.reshape([-1, img_size_flat])
N = X.shape[0]
K = 256
affinity = automatic_prunning
A = affinity(X)
Y = spectral_clustering(A, K)

Mat = numpy.zeros((256, 256))

for i in len(Y):
    Mat[lables[i]][Y[i]] += 1

total_num = 0.0
for j in 256:
    maxnum = 0
    for i in 256:
        if maxnum<Mat[i][j]:
            maxnum = Mat[i][j]
    total_num += maxnum

print(total_num/len(Y))