from itertools import permutations
from typing import Callable, List

import numpy as np


def monotonic(S: List[float]):
    """
    :param S: a set together with an index (given here, both together, as a list).
    """
    prev = S[0]
    for next in S[1:]:
        if next < prev:
            return False
        prev = next
    return True


def solve_sp(S: List[float]):
    """
    :param S: a set together with an index (given here, both together, as a list).
    """
    solutions = []
    indexes = permutations(S)
    for I in indexes:
        if monotonic(I):
            solutions.append(I)
    print(solutions)
    

def solve_sp_fast(S: List[float], generators: List[np.ndarray], compass: Callable):
    
    changed = True
    current_S = S
    current_m = compass(S)

    while changed:
        changed = False
        for gen in generators:
            new_S = current_S @ gen.T
            new_m = compass(new_S)
            if new_m < current_m:
                changed = True
                current_S = new_S
                current_m = new_m
                break

    print(current_S)
    
    
def compass_m(I):
    n = len(I)
    O = np.arange(n)
    return sum((O - I) ** 2)


def compass_d(I):
    m = 0
    for i in range(1, len(I)):
        m += (I[i] - I[i - 1]) ** 2
    return m