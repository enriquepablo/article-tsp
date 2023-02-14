from time import time
from typing import Callable, Dict, List, Tuple
import functools
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import models, layers
import tensorflow as tf

from sklearn.preprocessing import StandardScaler


def regular_points(n: int, radius: float = .5, translation: Tuple[float, float] = (.5, .5)) -> List[Tuple[float, float]]:
    """
    Generate `n` points regularly distributed in a circle of radius `radius`
    with center at `translation`.
    
    :param n: number of points
    :param radius: radius of circle
    :param translation: center of circle
    :return: list of points
    """
    one_segment = math.pi * 2 / n

    points = [
        (math.sin(one_segment * i) * radius,
         math.cos(one_segment * i) * radius)
        for i in range(n)]

    if translation:
        points = [[sum(pair) for pair in zip(point, translation)]
                  for point in points]

    return points


def regular_data(n: int, radius: float = .5, translation: Tuple[float, float] = (.5, .5)) -> Dict[str, np.ndarray]:
    """
    Use `regular_points` to produce data for a regular circuit.
    
    :param n: number of points
    :param radius: radius of circle
    :param translation: center of circle
    :return: Dictionary with keys:
        + positions: positions of the cities, pairs of numbers
        + circuit: names of the cities
    """
    points = regular_points(n, radius=radius, translation=translation)

    return {
        'positions': points,
        'circuit': list(range(n))
    }


def get_circle_matrix(n: int, matrix: np.ndarray) -> np.ndarray:
    """
    Use `regular_data` to produce data for a regular circuit
    that has `n` points and the aproximate size of the circuit in `matrix`.
    
    :param n: number of points
    :param matrix: matrix of complex positions
    :return: matrix of complex positions for a regular circuit
    """
    dmatrix = dmatrix_from_pmatrix(matrix)
    dmatrix.shape = (n**2,)
    radius = max(set(dmatrix))
    data = regular_data(n, radius=radius, translation=(0, 0))
    return pmatrix_from_data(data)


def quadrance(pmatrix: np.ndarray, circle: np.ndarray) -> float:
    """
    Measure the sqare of the distance from the circuit given by `pmatrix`
    to the circuit given by `circle`.
    
    :param pmatrix: matrix representing an arbitrary circuit
    :param circle: matrix representing a regular circuit
    :return: the quadrance between both
    """
    dif = (pmatrix - circle)
    qua = dif * dif.conj()
    return sum(sum(qua))


def get_random_data(n: int) -> Dict[str, np.ndarray]:
    """
    Produce a random instance of the TSP with `n` cities.

    :param n: The number of cities
    :return: Dictionary with keys:
        + positions: positions of the cities, pairs of numbers
        + circuit: names of the cities
    """
    return {
        'positions': np.random.random((n, 2)),
        'circuit': np.arange(n)
    }


def complex_positions_from_data(data: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Convert the positions key in the dictionary returned by `get_random_data`
    into a 1d array of complex numbers.

    :param data: data for a TSP instance, as provided by `get_random_data`
    :return: city positions as an array of complex numbers.
    """
    return np.array([complex(*p) for p in data['positions']])


def pmatrix_from_complex_positions(cpos: np.ndarray) -> np.ndarray:
    """
    Convert a 1d array of positions as complex numbers
    into an n x n matrix in which each row carries the complex positions
    centered on each different city.

    :param cpos: complex positions as returned by `complex_positions_from_data`.
    :return: matrix of complex positions
    """
    matrix = []
    for p in cpos:
        row = []
        for q in cpos:
            d = q - p
            row.append(d)
        matrix.append(row)
    return np.array(matrix)


def pmatrix_from_data(data: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Convert the positions key in the dictionary returned by `get_random_data`
    into an n x n matrix in which each row carries the complex positions
    centered on each different city.

    :param data: data for a TSP instance, as provided by `get_random_data`
    :return: matrix of complex positions
    """
    cpos = complex_positions_from_data(data)
    return pmatrix_from_complex_positions(cpos)


def paint_circuit(pmatrix: np.ndarray):
    """
    Plot the circuit correspoding to the provided matrix of complex positions

    :param pmatrix: matrix of complex positions
    """
    xs = pmatrix[0].real
    ys = pmatrix[0].imag
    d = length(pmatrix)
    plt.plot(xs, ys, '-', color='blue')
    plt.plot((xs[0], xs[-1]), (ys[0], ys[-1]), '-', color='blue')
    plt.title(f"Circuit length: {d:.4f}")
    plt.show()


def dmatrix_from_pmatrix(pmatrix: np.ndarray) -> np.ndarray:
    """
    Obtain a distances matrix from matrix of complex positions

    :param pmatrix: matrix of complex positions
    :return: distances matrix
    """
    return (pmatrix * np.conjugate(pmatrix)) ** (1 / 2)


@functools.cache
def roller_matrix(n: int) -> np.ndarray:
    """
    Construct a roller matrix

    :param n: the number of rows and columns
    :return: the roller matrix
    """
    return np.roll(np.eye(n), 1, axis=(1,))


def length(pmatrix: np.ndarray) -> np.ndarray:
    """
    Calculate the length of the circuit determined by
    the provided matrix of complex positions.

    :param pmatrix: a matrix of complex positions
    :return: the length of the circuit
    """
    roller = roller_matrix(len(pmatrix))
    return np.tensordot(abs(pmatrix), roller)


def path_to_sink(pmatrix: np.ndarray, gens: Callable, compass: Callable) -> List[np.ndarray]:
    """
    Generate a path, as a list of matrices of complex positions,
    given an initial matrix of complex positions,
    a generator set, and a compass function,
    so that each matrix in the list scores less than the previous
    at the compass function.

    :param pmatrix: a matrix of complex positions
    :param gens: function that will return an iterable of permutation matrices,
                 which together form a set of generators.
    :param compass: compass function
    :return: A list of matrices of complex positions, ending in a sink.
    """
    n = len(pmatrix)
    path = [pmatrix]
    w = compass(pmatrix)
    changed = True
    while changed:
        changed = False
        for gen in gens(n):
            new_pmatrix = gen @ pmatrix @ gen.T
            w2 = compass(new_pmatrix)
            if w2 < w:
                pmatrix = new_pmatrix
                w = w2
                changed = True
                path.append(pmatrix)
                break
    return path



def path_to_sink_from_circle(pmatrix: np.ndarray, gens: Callable) -> List[np.ndarray]:
    """
    Generate a path, as a list of matrices of complex positions,
    given an initial matrix of complex positions,
    and a generator set, using as (hardcoded) compass function
    the distance from the circuit to a regular circular circuit;
    so that each matrix in the list scores less than the previous
    at the comapss function.

    :param pmatrix: a matrix of complex positions
    :param gens: function that will return an iterable of permutation matrices,
                 which together form a set of generators.
    :return: A list of matrices of complex positions, ending in a sink.
    """
    n = len(pmatrix)
    circle = get_circle_matrix(n, pmatrix)
    w = quadrance(pmatrix, circle)
    path = [pmatrix]
    changed = True
    while changed:
        changed = False
        for gen in gens(n):
            new_matrix = gen @ pmatrix @ gen.T
            w2 = quadrance(new_matrix, circle)
            if w2 < w:
                pmatrix = new_matrix
                w = w2
                # print(w)
                changed = True
                path.append(pmatrix)
                break

    return path


def randomize_data(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Shuffle the data of a TSP instance to obtain a different circuit of the same points.

    :param data: data for a TSP instance, as provided by `get_random_data`
    :return: data for the same TSP instance, but with a different random circuit
    """
    idx = np.arange(len(data['circuit']))
    np.random.shuffle(idx)
    idx = list(idx)
    names = np.array(data['circuit'])[idx]
    pos = np.array(data['positions'])[idx]
    return {
        'circuit': names,
        'positions': pos
    }


def path_to_sink_from_data(data: Dict[str, np.ndarray], gens: Callable, compass: Callable) -> List[np.ndarray]:
    """
    Use `path_to_sink` to generate a path to a sink.

    :param data: data for a TSP instance, as provided by `get_random_data`
    :param gens: function that will return an iterable of permutation matrices,
                 which together form a set of generators.
    :param compass: compass function
    :return: A list of matrices of complex positions, ending in a sink.
    """
    matrix = pmatrix_from_data(data)
    return path_to_sink(matrix, gens, compass)


def path_to_sink_from_data_from_circle(data: Dict[str, np.ndarray], gens: Callable) -> List[np.ndarray]:
    """
    Use `path_to_sink_from_circle` to generate a path to a sink.

    :param data: data for a TSP instance, as provided by `get_random_data`
    :param gens: function that will return an iterable of permutation matrices,
                 which together form a set of generators.
    :return: A list of matrices of complex positions, ending in a sink.
    """
    matrix = pmatrix_from_data(data)
    return path_to_sink_from_circle(matrix, gens)


def local_lengths(dmatrix: np.ndarray) -> np.ndarray:
    """
    Return a 1d array with the circuit lengths as seen from the pow
    of each point, as 1d circuits.

    :param dmatrix: distances matrix
    :return: list of lengths
    """
    dm_r = np.roll(dmatrix, 1, axis=(1,))
    dif = dmatrix - dm_r
    sq = abs(dif)
    return np.sum(sq, axis=(1,))


def datapoint_from_pmatrix(pm: np.ndarray, gens: Callable) -> Dict[str, np.ndarray]:
    """
    Gather data corresponding to the node given by the provided matrix of complex positions

    :param pmatrix: a matrix of complex positions
    :param gens: function that will return an iterable of permutation matrices,
                 which together form a set of generators.
    :return: dictionary with the different kinds of data gathered
    """
    data = {}
    data['points'] = [(i.real, i.imag) for i in pm[0]]
    le = data['length'] = float(length(pm))
    ll = data['local_lengths'] = list(local_lengths(abs(pm)))
    data['diff_in'] = []
    data['diff_out'] = []
    data['local_diff_in'] = []
    data['local_diff_out'] = []

    for gen in gens(len(pm)):
        pmi = pm @ gen
        pmo = pm @ gen.T

        li = length(pmi)
        lli = local_lengths(abs(pmi))
        lo = length(pmo)
        llo = local_lengths(abs(pmo))

        data['diff_in'].append(float(li - le))
        data['diff_out'].append(float(lo - le))
        data['local_diff_in'].append(list(lli - ll))
        data['local_diff_out'].append(list(llo - ll))

    return data


def make_dataset(n: int, gens: Callable, gens_long: Callable, size: int, nsamples: int, setname: str = 'none'):
    """
    Make dataset of randomly generated data and save it to the filesystem.

    :param n: number of points (cities) in each data point
    :param gens: function that will return an iterable of permutation matrices,
                 which together form a set of generators.
    :param gens_long: function that will return an iterable of permutation matrices,
                 which together form a set of generators.
    :param size: number of datapoints to generate
    :param nsamples: number of attempts to find a sink for each data point.
    :param setname: string used to compose the filenames of each datapoint.
    """

    if not os.path.exists('datasets-raw'):
        os.mkdir('datasets-raw')

    name = os.path.join('datasets-raw', f"dataset-{n}-{setname}-{time()}")
    os.mkdir(name)
    logname = os.path.join(name, 'log')
    with open(logname, 'w') as log:

        for i in range(size):
            data = get_random_data(n)
            print('generated data')
            fails = []
            target = None

            pms = []
            for _ in range(nsamples):
                rdata = randomize_data(data)
                pm = path_to_sink_from_data(rdata, gens_long, length)[-1]
                pms.append(pm)

            pms.sort(key=lambda x: float(length(x)))
            log.write(f"{i}: {list(sorted([float(length(pm)) for pm in pms]))}\n")
            lengths = sorted(list(set([float(length(pm)) for pm in pms])))
            target = pms[0]

            prev = lengths[0]
            for lngth in lengths:
                if not np.isclose(prev, lngth):
                    for pm in pms[1:]:
                        if np.isclose(length(pm), lngth):
                            fails.append(pm)
                            prev = lngth
                            break

            print(f"Saving {len(fails)} fails")
            save_datapoint(target, fails, gens, name)


def save_datapoint(target: np.ndarray, fails: List[np.ndarray], gens: Callable, dir: str):
    """
    Save a data point to the filesystem.

    :param target: matrix of complex positions corresponding to the solution to the TSP instance
    :param fails: matrices of complex positions corresponding to sinks that are not the solution
    :param gens: function that will return an iterable of permutation matrices,
                 which together form a set of generators.
    :param dir: name of directory in which to store the data point.
    """
    data = {
        'target': datapoint_from_pmatrix(target, gens),
        'fails': [datapoint_from_pmatrix(f, gens) for f in fails],
    }
    datastr = json.dumps(data, indent=2)

    name = f"circuit-{data['target']['length']}.json"
    fname = os.path.join(dir, name)
    print(f"saving {fname}...")
    with open(fname, 'w') as f:
        f.write(datastr)


def load_dataset(dirname: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset from the filesystem.

    :param dirname: name of directory holding the dataset
    :return: array of arrays of data, one for each sink in each datapoint,
             and array of booleans, indicating whether each array of data corresponds to a solution or not.
    """
    contents = os.listdir(dirname)
    posset = []
    negset = []
    for name in contents:
        if not name.startswith('.'):
            fname = os.path.join(dirname, name)
            try:
                with open(fname) as f:
                    raw = json.loads(f.read())
            except json.JSONDecodeError:
                print(f"Problem reading {fname}")
                continue

            posset.append(load_datapoint(raw['target']))

            for fail in raw['fails']:

                negset.append(load_datapoint(fail))

    negset = np.array(negset)
    posset = np.array(posset)

    if len(negset) > len(posset):
        idx = np.random.choice(len(negset), len(posset))
        negset = negset[idx]

    posset = np.array(list(zip(posset, np.ones(len(posset)))))
    negset = np.array(list(zip(negset, np.zeros(len(negset)))))

    dataset = np.concatenate((posset, negset))
    np.random.shuffle(dataset)

    X = np.array(list(dataset[:, 0]))
    Y = np.array(list(dataset[:, 1]))

    return X, Y


def load_datapoint(dp: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Load data for a sink, including data for neighbouring nodes

    :param dp: data for a particular sink
    :return: array of numbers
    """
    data = []
    data.extend([x for dl in dp['local_diff_in'] for x in dl])
    data.extend([x for dl in dp['local_diff_out'] for x in dl])
    return np.array(data)


def make_model(dataset, optimizer='adam', activation='relu'):
    """
    Make ML model
    """
    X, Y = load_dataset(f"datasets/{dataset}/dataset-all")

    n_features = len(X[0])

    epochs = round(len(X) / 64)

    # Define the scaler
    scaler = StandardScaler().fit(X)

    # Scale the train set
    Xs = scaler.transform(X)

    model_layers = [
        layers.Dense(name="h1", input_dim=n_features,
                     units=int(round((n_features + 1) / 2)),
                     activation=activation),
        layers.Dropout(name="drop1", rate=0.2),

        layers.Dense(name="h2", units=8,
                     activation=activation),
        layers.Dropout(name="drop2", rate=0.2),

        layers.Dense(name="output", units=1, activation='sigmoid')]

    model = models.Sequential(name="DeepNN", layers=model_layers)

    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(x=Xs, y=Y, batch_size=64, epochs=epochs)

    return model, scaler


def check_model(model, scaler, dataset):
    X, Y = load_dataset(f"datasets/{dataset}/dataset-test")
    Xs = scaler.transform(X)

    model.evaluate(Xs, Y, verbose=1)

    predictions_t = model.predict(Xs)

    cm = tf.math.confusion_matrix(labels=Y, predictions=np.round(predictions_t))

    print(cm)
