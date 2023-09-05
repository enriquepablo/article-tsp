from itertools import combinations, permutations, product

import numpy as np
import matplotlib.pyplot as plt


def random_sp(n):
    return np.random.random(n)


def random_tsp(n):
    return np.squeeze(np.random.random((n, 2)).view(complex))


def phi(p):
    return sum((p - np.arange(len(p))) ** 2)


def psi(p):
    return sum(abs(p - np.roll(p, 1))**2)


def psi2(p):
    return sum(abs(p[:-1] - p[1:])**2)


def build_curve(p, G, measure):
    
    m = measure(p)
    curve = [p]
    
    changed = True
    while changed:
        changed = False
    
        next_p = None
        next_m = 0

        for g in G:
            maybe_p = p @ g
            maybe_m = measure(maybe_p)

            if maybe_m < m:
                if next_p is None or maybe_m < next_m:
                    changed = True
                    
                    next_p = maybe_p
                    next_m = maybe_m
                    
        if changed:
            p = next_p
            m = next_m
            curve.append(p)
        
    return curve


def build_curve2(p, G, measure, index=-1):
    
    m = measure(p)
    curve = [p]
    
    changed = True
    while changed:
        changed = False
    
        next_points = []

        for g in G:
            maybe_p = p @ g
            maybe_m = measure(maybe_p)
            

            if maybe_m < m:
                changed = True
                next_points.append((maybe_p, maybe_m))
                    
        if changed:
            next_points.sort(key=lambda x: x[1])
            p, m = next_points[index]
            curve.append(p)
        
    return curve


def nswap_generators(n):
    G = []
    eye = np.eye(n, dtype=np.uint16)
    for i in range(n):
        j = (i + 1) % n
        g = np.copy(eye)
        g[i] = eye[j]
        g[j] = eye[i]
        G.append(g)
    return G


def lswap_generators(n):
    G = []
    eye = np.eye(n, dtype=np.uint16)
    ijs = combinations(range(n), 2)
    for i, j in ijs:
        g = np.copy(eye)
        g[i] = eye[j]
        g[j] = eye[i]
        G.append(g)
    return G


def k_opt_generators(n, k):
    eye = np.eye(n, dtype=np.uint16)
    all_cuts = combinations(range(n), k)
    G = []
    seen = set()
    for cuts in all_cuts:
        offset = cuts[0]
        cuts = np.array(cuts[1:]) - offset
        pieces = []
        start = 0
        for cut in cuts:
            piece = eye[start:cut]
            pieces.append((piece, piece[::-1]))
            start = cut
        last_piece = eye[start:]
        pieces.append((last_piece, last_piece[::-1]))
        reorderings = permutations(pieces)
        for reordering in reorderings:
            orderings = product((0, 1), repeat=k)
            for ordering in orderings:
                newgen = [piece[o] for piece, o in zip(reordering, ordering)]
                anewgen = np.concatenate(newgen)
                maybe_newgen = np.eye(n, dtype=np.uint16)
                maybe_newgen[:offset] = anewgen[n - offset:]
                maybe_newgen[offset:] = anewgen[:n - offset]
                s = ''.join(map(str, np.ravel(maybe_newgen)))
                if s not in seen:
                    G.append(maybe_newgen)
                    seen.add(s)
    return G