import functools
import itertools

import numpy as np


@functools.cache
def nswap_generators(n):
    """
    return list containing n x n permutation matrices corresponding to the
    generator set given by swapping any 2 consecutive elements
    """
    eye = np.eye(n, dtype=int)
    gens = [eye]
    for i in range(n):
        gens.append(get_nswap_generator(eye, n, i))
    return gens


def get_nswap_generator(eye, n, i):
    """
    return n x n permutation matrix that will swap the ith element with the
    (i + 1)th element, or, if the ith is the last element,
    the ith with the 1st.
    """
    perm = np.eye(n, dtype=int)
    j = (i + 1) % n
    perm[i] = eye[j]
    perm[j] = eye[i]
    return perm


@functools.cache
def lswap_generators(n):
    """
    return list containing n x n permutation matrices corresponding to the
    generator set given by swapping any 2 elements
    """
    eye = np.eye(n, dtype=int)
    gens = [eye]
    for i in range(n):
        for j in range(i, n):
            perm = get_lswap_generator(eye, n, i, j)
            if perm is not None:
                gens.append(perm)
    return gens


def get_lswap_generator(eye, n, i, j):
    """
    return n x n permutation matrix that will swap the ith element with the jth.
    """
    perm = None
    if i != j:
        perm = np.eye(n, dtype=int)
        perm[j] = eye[i]
        perm[i] = eye[j]
    return perm


@functools.cache
def insertion_generators(n):
    """
    return list containing n x n permutation matrices corresponding to the
    generator set given by moving any element to any place.
    """
    eye = np.eye(n, dtype=int)
    gens = [eye]
    for i in range(n):
        for j in range(n):
            perm = get_insertion_generator(eye, n, i, j)
            if perm is not None:
                gens.append(perm)
    return gens


def get_insertion_generator(eye, n, i, j):
    """
    return n x n permutation matrix that will move the ith element
    to the jth place
    """
    perm = None
    if i > j:
        perm = np.eye(n, dtype=int)
        perm[j] = eye[i]
        perm[j + 1: i + 1] = eye[j: i]
    elif j > i:
        perm = np.eye(n, dtype=int)
        perm[i: j] = eye[i + 1: j + 1]
        perm[j] = eye[i]
    return perm


@functools.cache
def get_gen4(n, i, j, k, m):
    """
    return n x n permutation matrix that will swap the i - j segment
    with the k - m segment.
    """
    eye = cache_eye(n)
    gen = np.copy(eye)
    gen[i: i + m - k] = eye[k: m]
    gen[i + m - k: i + m - j] = eye[j: k]
    gen[i + m - j: m] = eye[i: j]
    return gen


@functools.cache
def get_gen4_r1(n, i, j, k, m):
    """
    return n x n permutation matrix that will swap the i - j segment
    with the k - m segment, but reverting them before inserting them.
    """
    eye = cache_eye(n)
    gen = np.copy(eye)
    gen[i: i + m - k] = eye[k: m][::-1]
    gen[i + m - k: i + m - j] = eye[j: k]
    gen[i + m - j: m] = eye[i: j][::-1]
    return gen


@functools.cache
def get_gen4_r2(n, i, j, k, m):
    """
    return n x n permutation matrix that will revert both the i - j segment
    and the k - m segment.
    """
    eye = cache_eye(n)
    gen = np.copy(eye)
    gen[i: j] = eye[i: j][::-1]
    gen[k: m] = eye[k: m][::-1]
    return gen


def gen4s(n):
    """
    generate the permutation matrices corresponding to the generator set
    determined by the functions in this module that need 4 arguments to
    produce a permutation matrix.
    """
    breaks = itertools.combinations(range(n), 4)
    for br in breaks:
        yield get_gen4(n, *br)
        yield get_gen4_r1(n, *br)
        yield get_gen4_r2(n, *br)


@functools.cache
def get_gen3(n, i, j, k):
    """
    return n x n permutation matrix that will swap the i - j segment
    with the j - k segment.
    """
    eye = cache_eye(n)
    gen = np.copy(eye)
    gen[i: i + k - j] = eye[j: k]
    gen[i + k - j: k] = eye[i: j]
    return gen


@functools.cache
def get_gen3_r1(n, i, j, k):
    """
    return n x n permutation matrix that will swap the i - j segment
    with the j - k segment, but reverting the j - k segment.
    """
    eye = cache_eye(n)
    gen = np.copy(eye)
    gen[i: i + k - j] = eye[j: k][::-1]
    gen[i + k - j: k] = eye[i: j]
    return gen


@functools.cache
def get_gen3_r2(n, i, j, k):
    """
    return n x n permutation matrix that will swap the i - j segment
    with the j - k segment, but reverting the i - j segment.
    """
    eye = cache_eye(n)
    gen = np.copy(eye)
    gen[i: i + k - j] = eye[j: k]
    gen[i + k - j: k] = eye[i: j][::-1]
    return gen


@functools.cache
def get_gen3_r3(n, i, j, k):
    """
    return n x n permutation matrix that will swap the i - j segment
    with the j - k segment, but reverting both after joining them.
    """
    eye = cache_eye(n)
    gen = np.copy(eye)
    tmp = np.concatenate((eye[j: k], eye[i: j]))
    gen[i: k] = tmp[::-1]
    return gen


def gen3s(n):
    """
    generate the permutation matrices corresponding to the generator set
    determined by the functions in this module that need 3 arguments to
    produce a permutation matrix.
    """
    breaks = itertools.combinations(range(n), 3)
    for br in breaks:
        for get_gen in (get_gen3, get_gen3_r1, get_gen3_r2, get_gen3_r3):  # get_gen3,
            yield get_gen(n, *br)


@functools.cache
def cache_eye(n):
    return np.eye(n, dtype=int)


@functools.cache
def get_gen2(n, i, j):
    """
    return n x n permutation matrix that will revert the i - j segment.
    """
    eye = cache_eye(n)
    gen = np.copy(eye)
    gen[i: j] = eye[i: j][::-1]
    return gen


def gen2s(n):
    """
    generate the permutation matrices corresponding to the generator set
    determined by the functions in this module that need 2 arguments to
    produce a permutation matrix.
    """
    breaks = itertools.combinations(range(n), 2)
    for br in breaks:
        yield get_gen2(n, *br)


def gen23s(n):
    """
    generate the permutation matrices corresponding to the generator set
    determined by the functions in this module that need 2 or 3 arguments to
    produce a permutation matrix.
    """
    for gen in gen2s(n):
        yield gen
    for gen in gen3s(n):
        yield gen


def gens(n):
    """
    generate the permutation matrices corresponding to the generator set
    determined by the functions in this module that need 2, 3 or 4 arguments to
    produce a permutation matrix.
    """
    for gen in gen2s(n):
        yield gen
    for gen in gen3s(n):
        yield gen
    for gen in gen4s(n):
        yield gen
