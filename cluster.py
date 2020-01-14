import functools

import math
import collections
import numpy as np
from scipy.sparse import coo_matrix
from math import log
from numpy import bincount


compare = lambda x, y: collections.Counter(x) == collections.Counter(y)


class Point:
    def __init__(self, coords, cheat=None, reference=None, group_name=None, fb_url=None):
        self.coords = coords
        self.n = len(coords)
        self.reference = reference
        self.avrage_distance = 0
        self.max_distance = 0
        self.silhuette = None
        self.cheat = cheat
        self.purity = None
        self.nmi = 0
        self.distances = []
        self.group_name = group_name
        self.fb_url = fb_url

    def __repr__(self):
        return str(self.reference)+": "+str(self.coords)

    def getDistance(self, b):
        if self.n != b.n: raise Exception("ILLEGAL: non comparable points")
        ret = functools.reduce(lambda x,y: x + pow((self.coords[y]-b.coords[y]), 2),range(self.n),0.0)
        return (ret)
        #return reduce(lambda x,y: x+(self.coords[y]-b.coords[y]),range(self.n),0.0)
    def __eq__(self, other):
        if hasattr(other, 'coords') and hasattr(other, 'n'):
            return self.coords == other.coords and self.n == other.n
        else:
            return False;
class Cluster:
    def __init__(self, idClustra):
        self.points = []
        self.primeri = []
        self.clusterId = idClustra
        self.max_distance = 0
        self.name = None
        self.reference = None
        self.n = 1
        self.weight = [0]*self.n
        self.purity = None
        self.idx1 = None
        self.idx2 = None
        self.centroid = None
        self.dim = None

    def __repr__(self):
        return str(self.clusterId)
    def represent(self):
        return self.centroid
    def calculateCentroid(self):
        """
        Calculate the new cluster centroid
        """
        reduce_coord = lambda i:functools.reduce(lambda x,p : x + p.coords[i],self.points,0.0)
        if len(self.points) <= 0:
            centroid_coords = [0 for i in range(self.dim)]
        else:
            centroid_coords = [reduce_coord(i)/len(self.points) for i in range(self.dim)]
            centroid_coords = [round(elem, 2) for elem in centroid_coords]
        return Point(centroid_coords, None, self.name)

    def update(self, id1, id2, dist, tocke):
        self.idx1 = id1
        self.idx2 = id2
        self.distance = dist
        for points in tocke:
            for p in points:
                self.points.append(p)
                self.primeri.append(p)
        self.n = len(self.primeri)
    def stats(self):
        sum_d = 0
        n=0
        for p in self.points:
            n+=1
            distance = p.getDistance(self.centroid)
            sum_d+=distance
            if distance > self.max_distance:
                self.max_distance = distance
        if n == 0:
            self.avrage_distance = 0
            return
        self.avrage_distance = sum_d/n



"""
SCIPY CODE
"""

def normalized_mutual_info_score(labels_true, labels_pred):
    """Normalized Mutual Information between two clusterings
    Normalized Mutual Information (NMI) is an normalization of the Mutual
    Information (MI) score to scale the results between 0 (no mutual
    information) and 1 (perfect correlation). In this function, mutual
    information is normalized by ``sqrt(H(labels_true) * H(labels_pred))``
    This measure is not adjusted for chance. Therefore
    :func:`adjusted_mustual_info_score` might be preferred.
    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.
    This metric is furthermore symmetric: switching ``label_true`` with
    ``label_pred`` will return the same score value. This can be useful to
    measure the agreement of two independent label assignments strategies
    on the same dataset when the real ground truth is not known.
    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        A clustering of the data into disjoint subsets.
    labels_pred : array, shape = [n_samples]
        A clustering of the data into disjoint subsets.
    Returns
    -------
    nmi: float
       score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling
    See also
    --------
    adjusted_rand_score: Adjusted Rand Index
    adjusted_mutual_info_score: Adjusted Mutual Information (adjusted
        against chance)
    Examples
    --------
    Perfect labelings are both homogeneous and complete, hence have
    score 1.0::
      >>> from sklearn.metrics.cluster import normalized_mutual_info_score
      >>> normalized_mutual_info_score([0, 0, 1, 1], [0, 0, 1, 1])
      1.0
      >>> normalized_mutual_info_score([0, 0, 1, 1], [1, 1, 0, 0])
      1.0
    If classes members are completely split across different clusters,
    the assignment is totally in-complete, hence the NMI is null::
      >>> normalized_mutual_info_score([0, 0, 0, 0], [0, 1, 2, 3])
      0.0
    """
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    # Special limit cases: no clustering since the data is not split.
    # This is a perfect match hence return 1.0.
    if (classes.shape[0] == clusters.shape[0] == 1
            or classes.shape[0] == clusters.shape[0] == 0):
        return 1.0
    contingency = contingency_matrix(labels_true, labels_pred)
    contingency = np.array(contingency, dtype='float')
    # Calculate the MI for the two clusterings
    mi = mutual_info_score(labels_true, labels_pred,
                           contingency=contingency)
    # Calculate the expected value for the mutual information
    # Calculate entropy for each labeling
    h_true, h_pred = entropy(labels_true), entropy(labels_pred)
    nmi = mi / max(np.sqrt(h_true * h_pred), 1e-10)
    return nmi


def adjusted_rand_score(labels_true, labels_pred):
    """Rand index adjusted for chance
    The Rand Index computes a similarity measure between two clusterings
    by considering all pairs of samples and counting pairs that are
    assigned in the same or different clusters in the predicted and
    true clusterings.
    The raw RI score is then "adjusted for chance" into the ARI score
    using the following scheme::
        ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)
    The adjusted Rand index is thus ensured to have a value close to
    0.0 for random labeling independently of the number of clusters and
    samples and exactly 1.0 when the clusterings are identical (up to
    a permutation).
    ARI is a symmetric measure::
        adjusted_rand_score(a, b) == adjusted_rand_score(b, a)
    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        Ground truth class labels to be used as a reference
    labels_pred : array, shape = [n_samples]
        Cluster labels to evaluate
    Returns
    -------
    ari: float
       Similarity score between -1.0 and 1.0. Random labelings have an ARI
       close to 0.0. 1.0 stands for perfect match.
    Examples
    --------
    Perfectly maching labelings have a score of 1 even
      >>> from sklearn.metrics.cluster import adjusted_rand_score
      >>> adjusted_rand_score([0, 0, 1, 1], [0, 0, 1, 1])
      1.0
      >>> adjusted_rand_score([0, 0, 1, 1], [1, 1, 0, 0])
      1.0
    Labelings that assign all classes members to the same clusters
    are complete be not always pure, hence penalized::
      >>> adjusted_rand_score([0, 0, 1, 2], [0, 0, 1, 1])  # doctest: +ELLIPSIS
      0.57...
    ARI is symmetric, so labelings that have pure clusters with members
    coming from the same classes but unnecessary splits are penalized::
      >>> adjusted_rand_score([0, 0, 1, 1], [0, 0, 1, 2])  # doctest: +ELLIPSIS
      0.57...
    If classes members are completely split across different clusters, the
    assignment is totally incomplete, hence the ARI is very low::
      >>> adjusted_rand_score([0, 0, 0, 0], [0, 1, 2, 3])
      0.0
    References
    ----------
    .. [Hubert1985] `L. Hubert and P. Arabie, Comparing Partitions,
      Journal of Classification 1985`
      http://www.springerlink.com/content/x64124718341j1j0/
    .. [wk] http://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index
    See also
    --------
    adjusted_mutual_info_score: Adjusted Mutual Information
    """
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    n_samples = labels_true.shape[0]
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    # Special limit cases: no clustering since the data is not split;
    # or trivial clustering where each document is assigned a unique cluster.
    # These are perfect matches hence return 1.0.
    if (classes.shape[0] == clusters.shape[0] == 1
            or classes.shape[0] == clusters.shape[0] == 0
            or classes.shape[0] == clusters.shape[0] == len(labels_true)):
        return 1.0

    contingency = contingency_matrix(labels_true, labels_pred)

    # Compute the ARI using the contingency data
    sum_comb_c = sum(comb2(n_c) for n_c in contingency.sum(axis=1))
    sum_comb_k = sum(comb2(n_k) for n_k in contingency.sum(axis=0))

    sum_comb = sum(comb2(n_ij) for n_ij in contingency.flatten())
    prod_comb = (sum_comb_c * sum_comb_k) / float(comb(n_samples, 2, exact=1))
    mean_comb = (sum_comb_k + sum_comb_c) / 2.
    return ((sum_comb - prod_comb) / (mean_comb - prod_comb))

def check_clusterings(labels_true, labels_pred):
    """Check that the two clusterings matching 1D integer arrays"""
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    # input checks
    if labels_true.ndim != 1:
        raise ValueError(
            "labels_true must be 1D: shape is %r" % (labels_true.shape,))
    if labels_pred.ndim != 1:
        raise ValueError(
            "labels_pred must be 1D: shape is %r" % (labels_pred.shape,))
    if labels_true.shape != labels_pred.shape:
        raise ValueError(
            "labels_true and labels_pred must have same size, got %d and %d"
            % (labels_true.shape[0], labels_pred.shape[0]))
    return labels_true, labels_pred


def contingency_matrix(labels_true, labels_pred, eps=None):
    """Build a contengency matrix describing the relationship between labels.
    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        Ground truth class labels to be used as a reference
    labels_pred : array, shape = [n_samples]
        Cluster labels to evaluate
    eps: None or float
        If a float, that value is added to all values in the contingency
        matrix. This helps to stop NaN propagation.
        If ``None``, nothing is adjusted.
    Returns
    -------
    contingency: array, shape=[n_classes_true, n_classes_pred]
        Matrix :math:`C` such that :math:`C_{i, j}` is the number of samples in
        true class :math:`i` and in predicted class :math:`j`. If
        ``eps is None``, the dtype of this array will be integer. If ``eps`` is
        given, the dtype will be float.
    """
    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    # Using coo_matrix to accelerate simple histogram calculation,
    # i.e. bins are consecutive integers
    # Currently, coo_matrix is faster than histogram2d for simple cases
    contingency = coo_matrix((np.ones(class_idx.shape[0]),
                              (class_idx, cluster_idx)),
                             shape=(n_classes, n_clusters),
                             dtype=np.int).toarray()
    if eps is not None:
        # don't use += as contingency is integer
        contingency = contingency + eps
    return contingency


def mutual_info_score(labels_true, labels_pred, contingency=None):
    """Mutual Information between two clusterings
    The Mutual Information is a measure of the similarity between two labels of
    the same data. Where :math:`P(i)` is the probability of a random sample
    occurring in cluster :math:`U_i` and :math:`P'(j)` is the probability of a
    random sample occurring in cluster :math:`V_j`, the Mutual Information
    between clusterings :math:`U` and :math:`V` is given as:
    .. math::
        MI(U,V)=\sum_{i=1}^R \sum_{j=1}^C P(i,j)\log\\frac{P(i,j)}{P(i)P'(j)}
    This is equal to the Kullback-Leibler divergence of the joint distribution
    with the product distribution of the marginals.
    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.
    This metric is furthermore symmetric: switching ``label_true`` with
    ``label_pred`` will return the same score value. This can be useful to
    measure the agreement of two independent label assignments strategies
    on the same dataset when the real ground truth is not known.
    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        A clustering of the data into disjoint subsets.
    labels_pred : array, shape = [n_samples]
        A clustering of the data into disjoint subsets.
    contingency: None or array, shape = [n_classes_true, n_classes_pred]
        A contingency matrix given by the :func:`contingency_matrix` function.
        If value is ``None``, it will be computed, otherwise the given value is
        used, with ``labels_true`` and ``labels_pred`` ignored.
    Returns
    -------
    mi: float
       Mutual information, a non-negative value
    See also
    --------
    adjusted_mutual_info_score: Adjusted against chance Mutual Information
    normalized_mutual_info_score: Normalized Mutual Information
    """
    if contingency is None:
        labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
        contingency = contingency_matrix(labels_true, labels_pred)
    contingency = np.array(contingency, dtype='float')
    contingency_sum = np.sum(contingency)
    pi = np.sum(contingency, axis=1)
    pj = np.sum(contingency, axis=0)
    outer = np.outer(pi, pj)
    nnz = contingency != 0.0
    # normalized contingency
    contingency_nm = contingency[nnz]
    log_contingency_nm = np.log(contingency_nm)
    contingency_nm /= contingency_sum
    # log(a / b) should be calculated as log(a) - log(b) for
    # possible loss of precision
    log_outer = -np.log(outer[nnz]) + log(pi.sum()) + log(pj.sum())
    mi = (contingency_nm * (log_contingency_nm - log(contingency_sum))
          + contingency_nm * log_outer)
    return mi.sum()


def entropy(labels):
    """Calculates the entropy for a labeling."""
    if len(labels) == 0:
        return 1.0
    label_idx = np.unique(labels, return_inverse=True)[1]
    pi = bincount(label_idx).astype(np.float)
    pi = pi[pi > 0]
    pi_sum = np.sum(pi)
    # log(a / b) should be calculated as log(a) - log(b) for
    # possible loss of precision
    return -np.sum((pi / pi_sum) * (np.log(pi) - log(pi_sum)))

def comb2(n):
    # the exact version is faster for k == 2: use it by default globally in
    # this module instead of the float approximate variant
    return comb(n, 2, exact=1)

def comb(N, k, exact=False, repetition=False):
    """
    The number of combinations of N things taken k at a time.
    This is often expressed as "N choose k".
    Parameters
    ----------
    N : int, ndarray
        Number of things.
    k : int, ndarray
        Number of elements taken.
    exact : bool, optional
        If `exact` is False, then floating point precision is used, otherwise
        exact long integer is computed.
    repetition : bool, optional
        If `repetition` is True, then the number of combinations with
        repetition is computed.
    Returns
    -------
    val : int, ndarray
        The total number of combinations.
    Notes
    -----
    - Array arguments accepted only for exact=False case.
    - If k > N, N < 0, or k < 0, then a 0 is returned.
    Examples
    --------
    >>> k = np.array([3, 4])
    >>> n = np.array([10, 10])
    >>> sc.comb(n, k, exact=False)
    array([ 120.,  210.])
    >>> sc.comb(10, 3, exact=True)
    120L
    >>> sc.comb(10, 3, exact=True, repetition=True)
    220L
    """
    if repetition:
        return comb(N + k - 1, k, exact)
    if exact:
        N = int(N)
        k = int(k)
        if (k > N) or (N < 0) or (k < 0):
            return 0
        val = 1
        for j in range(min(k, N-k)):
            val = (val*(N-j))//(j+1)
        return val