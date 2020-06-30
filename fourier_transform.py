'''
Anibal M. Medina-Mardones, 2020
ammedmar@gmail.com

Code to perform the Fourier transform of higher dimensional signals.
More specifically, give the average connectome of a set of patients
we construct the weighted clique complex, with weights constructed in
three different ways, and transform the higher EGG-information signals
into the Fourier basis.
'''

import numpy as np
import scipy.io
from itertools import combinations
from scipy.special import comb
import os
import warnings
warnings.filterwarnings('ignore', message='divide by zero')
from time import perf_counter
import random


def import_data(patient, d=3):
    '''for each patient numbered by a string between 001 to 164 gives
    a dict with the Oinfo, Sinfo, total correlation and dual TC'''
    filename = os.path.join(
        folder, 'OinfoCopulasN{}/Oinfop{}.mat'.format(d + 1, patient))
    data = scipy.io.loadmat(filename)

    del data['__header__']
    del data['__version__']
    del data['__globals__']

    data['TC'] = (data['Sinfo'] + data['Oinfo']) / 2
    data['DTC'] = (data['Sinfo'] - data['Oinfo']) / 2

    return data


def construct_edge_weights():
    '''Average over patients in the form of a dict with keys pairs (i,j)
    with 1 < i < j < 20'''

    filename = os.path.join(folder, 'connectome/sc_norm_20rois.mat')
    conn = scipy.io.loadmat(filename)
    SC = conn['sc'].sum(axis=2) / 161

    if gaussian_weights:
        s = SC.std()
        m = SC.mean()
        np.random.seed(1)
        SC = np.random.normal(loc=m, scale=s, size=(20, 20))
        SC = np.absolute(SC)

    edge_weights = {}
    for i in range(20):
        for j in range(i + 1, 20):
            edge_weights[(i + 1, j + 1)] = SC[i, j]

    return edge_weights


def construct_simplex_weights(d, r=19, scheme='av'):
    '''returns the dict with items simplex: weight for each simplex of dim d
    options: min, max, av for determining the weight of a simplex based on the
    weight of its edges. If randomize is True then we randomly mix the pairs.
    '''
    edge_weights = construct_edge_weights()

    if scheme == 'min':
        simplex_weights = {
            spx: min(edge_weights[edge] for edge in combinations(spx, 2))
            for spx in tuple(combinations(range(1, r + 2), d + 1))}

    elif scheme == 'max':
        simplex_weights = {
            spx: max(edge_weights[edge] for edge in combinations(spx, 2))
            for spx in tuple(combinations(range(1, r + 2), d + 1))}

    elif scheme == 'av':
        simplex_weights = {
            spx: sum(edge_weights[edge]
                     for edge in combinations(spx, 2)) / comb(d + 1, 2)
            for spx in tuple(combinations(range(1, r + 2), d + 1))}

    if randomize:
        shuffled = list(simplex_weights.values())
        random.seed(4)
        random.shuffle(shuffled)
        simplex_weights = {spx: weight for spx,
                           weight in zip(simplex_weights.keys(), shuffled)}

    return simplex_weights


def thresholding(d, scheme='av', threshold=0):
    '''returns the list, to be used as a mask, of indices of simplices
    with weigth  > threshold'''
    simplex_weights = construct_simplex_weights(d, scheme=scheme)
    return [i for i, v in enumerate(simplex_weights.values()) if v > threshold]


def construct_weight_matrix(d, inverse=False):
    '''returns a diagonal matrix with the weights of simplices'''
    simplex_weights = construct_simplex_weights(d)
    num_rows = len(simplex_weights)
    weights = np.array(list(simplex_weights.values()), dtype=np.float)
    if inverse:
        weights = np.reciprocal(weights)
    weights = np.reshape(weights, (num_rows,))
    return scipy.sparse.csr_matrix(
        (weights, (range(num_rows), range(num_rows))),
        shape=(num_rows, num_rows), dtype=float)


def construct_boundary(d, r=19):
    '''constructs a sparse matrix representing the boundary map from chains of
    degree d to chains of degree d-1'''
    domain_basis = tuple(combinations(range(r + 1), d + 1))
    target_basis = tuple(combinations(range(r + 1), d))
    target_basis_ix = {tuple(v): index for index, v in enumerate(target_basis)}
    N = comb(r + 1, d + 1, exact=True)
    M = comb(r + 1, d, exact=True)
    D = scipy.sparse.csr_matrix((M, N), dtype=np.int8)
    for j in range(d + 1):
        jth_faces_ix = [
            target_basis_ix[tuple(np.concatenate((s[:j], s[j + 1:])))]
            for s in domain_basis]
        D += scipy.sparse.csr_matrix(
            ([(-1)**j] * N, (jth_faces_ix, range(N))),
            shape=(M, N), dtype=np.int8)
    return D


def construct_coboundary(d, r=19):
    '''the transpose of the boundary matrix'''
    return construct_boundary(d + 1, r).T


def construct_laplacian(d, scheme='av', threshold=0):
    '''constructs the weighted laplacian L = Lu + Ld'''
    nz = {i: thresholding(i, scheme=scheme, threshold=threshold)
          for i in [d - 1, d, d + 1]}

    # Lu = WIa Bp Wp BTa
    BTa = construct_coboundary(d)[:, nz[d]][nz[d + 1], :]
    Wp = construct_weight_matrix(d + 1)[:, nz[d + 1]][nz[d + 1], :]
    Bp = construct_boundary(d + 1)[:, nz[d + 1]][nz[d], :]
    WIa = construct_weight_matrix(d, inverse=True)[:, nz[d]][nz[d], :]

    Lu = np.multiply(WIa, np.multiply(Bp, np.multiply(Wp, BTa)))

    # Ld = BTm WIm Ba Wa
    Wa = construct_weight_matrix(d)[:, nz[d]][nz[d], :]
    Ba = construct_boundary(d)[:, nz[d]][nz[d - 1], :]
    WIm = construct_weight_matrix(
        d - 1, inverse=True)[:, nz[d - 1]][nz[d - 1], :]
    BTm = construct_coboundary(d - 1)[:, nz[d - 1]][nz[d], :]

    Ld = np.multiply(BTm, np.multiply(WIm, np.multiply(Ba, Wa)))

    return (Ld + Lu)


# _____________________________Parameters_____________________________


# directory where the data is
folder = '/Users/anibal/Google Drive/information_cochains'

# dimensions considered in the computation, values between 3 an 19
dimensions = [2]  # [2, 3]

# schemes to define the weight of a simplex based on the weight of its edges
schemes = ['av']  # ['av', 'min', 'max']

# to set equal to 0 weights less than the threshold
thresholds = [0.01]  # [0.01, 0.001, 0.0001]

# number of stored eigenvectors ordered by the size of its eigenvalue
stored_eigenvectors = 10

# types of information used for the signals
infos = ['Oinfo']  # ['Oinfo', 'Sinfo', 'TC', 'DTC']

# randomly pair (simplex: weight)
randomize = False

# normally distributed connectome weights
gaussian_weights = True

# names of patients from 001 to 164
patients = ([f'00{i}' for i in range(1, 10)] +
            [f'0{i}' for i in range(10, 100)] +
            [f'{i}' for i in range(100, 165)])


# output structure:

# dim
#   threshold_scheme
#     eigendata
#       eigenvalues
#       eigenvectors
#     info
#       patient

# _____________________________Main___________________________________________


for d in dimensions:
    os.mkdir(f'dim{d}')
    for threshold in thresholds:
        for scheme in schemes:
            time0 = perf_counter()
            L = construct_laplacian(d, scheme=scheme, threshold=threshold)
            time1 = perf_counter()
            eigenvalues, eigenvectors = np.linalg.eig(L.todense())
            time2 = perf_counter()
            os.makedirs(f'dim{d}/{scheme}_thresh{threshold}/eigendata')
            np.save(
                f'dim{d}/{scheme}_thresh{threshold}/eigendata/eigenvalues.npy',
                eigenvalues)
            np.save(
                f'dim{d}/{scheme}_thresh{threshold}/eigendata/eigenvectors.npy',
                np.argsort(eigenvalues)[:stored_eigenvectors])

            # fourier transforming the information cochains

            mask = thresholding(d, scheme=scheme, threshold=threshold)

            for info in infos:
                os.makedirs(f'dim{d}/{scheme}_thresh{threshold}/{info}')
                B = np.empty((L.shape[0], len(patients)))
                for idx, patient in enumerate(patients):
                    data = import_data(patient, d)
                    cochain = data[info][mask, :]
                    B[:, idx: idx + 1] = cochain

                X = np.linalg.solve(eigenvectors, B)
                time3 = perf_counter()

                print(f"dim {d} - threshold: {threshold} - scheme: {scheme} - info: {info} - " +
                      f"randomized pairing: {randomize} - gaussian weights: {gaussian_weights}")
                print('construct:', time1 - time0, 'diagonalize:',
                      time2 - time1, 'solve all:', time3 - time2)

                for idx, x in enumerate(X.T):
                    patient = patients[idx]
                    np.save(
                        f'dim{d}/{scheme}_thresh{threshold}/{info}/{d}harmonic{patient}.npy', x)
