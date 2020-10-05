
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
    prod_comb = (sum_comb_c * sum_comb_k) / float(comb(n_samples, 2))
    mean_comb = (sum_comb_k + sum_comb_c) / 2.
    return ((sum_comb - prod_comb) / (mean_comb - prod_comb))