from typing import Callable, Dict, Tuple
import numpy as np
import sklearn.datasets as ds
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from scipy.optimize import minimize
import tracemalloc
import datetime
import csv
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline

def generate_dataset(samples: int =1000, attributes: int =1, noise: int =42) -> Tuple[np.ndarray, np.ndarray]:
    return ds.make_regression(n_samples=samples, n_features=attributes, noise=noise)

def normalize_data_set(train: np.ndarray, test: np.ndarray =None) -> Tuple[np.ndarray, np.ndarray]:
    scaler = MinMaxScaler()
    scaler.fit(train)
    return (scaler.transform(train), scaler.transform(test) if test is not None else None)

def benchmark(fun: Callable[[np.ndarray, np.ndarray], None], x: np.ndarray, y: np.ndarray, n: int =200) -> Dict[str, Dict[str,float]]:
    assert n > 0
    start = datetime.datetime.now()
    tracemalloc.start()

    [fun(x, y) for _ in range(n)]

    _ = tracemalloc.take_snapshot()
    current, peak = tracemalloc.get_traced_memory()
    mem = tracemalloc.get_tracemalloc_memory()
    tracemalloc.stop()
    diff = datetime.datetime.now() - start

    return {"memory": {"total": peak, "mean": peak/n}, "time": {"total": diff.microseconds, "mean":diff.microseconds/n}}
X, Y = generate_dataset(1012)


class AnalyticLinearRegressor:
    
    def __init__(self) -> None:
        self.__theta: np.ndarray =None     
        self.__is_fit: bool =False

    def fit(self, train: np.ndarray, targets: np.ndarray) -> None:
        broadcast_train = np.ones((train.shape[0], train.shape[1] + 1))
        broadcast_train[:, 1:] = train
        self.__theta = np.linalg.inv(broadcast_train.T @ broadcast_train) @ broadcast_train.T @ targets
        self.__is_fit = True
    
    def predict_batch(self, samples: np.ndarray) -> np.ndarray:
        if not self.__is_fit:
            raise Exception("Model has not been fit yet!")
        broadcast_samples = np.ones((samples.shape[0], samples.shape[1] + 1))
        broadcast_samples[:, 1:] = samples
        return broadcast_samples @ self.__theta

    def predict(self, sample: np.ndarray) -> np.ndarray:
        if not self.__is_fit:
            raise Exception("Model has not been fit yet!")
        broadcast_sample = np.ones((sample.shape[0], sample.shape[1] + 1))
        broadcast_sample[:, 1:] = sample
        return broadcast_sample @ self.__theta

class NumericLinearRegressor:
    
    def __init__(self) -> None:
        self.__theta: np.ndarray =None     
        self.__is_fit: bool =False
        self.__is_fitting: bool =False

    def fit(self, train: np.ndarray, targets: np.ndarray) -> None:
        self.__is_fitting = True
        self.__is_fit = False
        N = targets.shape[0]
        _theta = np.ones(train.shape[1] + 1)

        def loss(__theta: np.ndarray) -> float:
            predicted: np.ndarray = self.predict_batch(train, __theta)
            diff: np.ndarray = (targets - predicted)
            return (1/N) * (diff.T @ diff)

        self.__theta = minimize(fun=loss, x0=_theta, method='Powell')
        self.__is_fit = True
        self.__is_fitting = False
    
    def predict_batch(self, samples: np.ndarray, theta: np.ndarray =None) -> np.ndarray:
        if not (self.__is_fit or self.__is_fitting):
            raise Exception("Model has not been fit yet!")
        broadcast_samples = np.ones((samples.shape[0], samples.shape[1] + 1))
        if theta is None:
            theta = self.__theta
        broadcast_samples[:, 1:] = samples
        return broadcast_samples @ theta

    def predict(self, sample: np.ndarray) -> np.ndarray:
        if not (self.__is_fit or self.__is_fitting):
            raise Exception("Model has not been fit yet!")
        broadcast_sample = np.ones((sample.shape[0], sample.shape[1] + 1))
        broadcast_sample[:, 1:] = sample
        return broadcast_sample @ self.__theta

regressor = AnalyticLinearRegressor()
regressor.fit(X, Y)
agressor = NumericLinearRegressor()
agressor.fit(X, Y)
# print(benchmark(lambda x,y: regressor.fit(x, y), X, Y, 1000))
# print(benchmark(lambda x,y: agressor.fit(x, y), X, Y, 1000))

with open("./Regression/task2_dataset_1.csv", 'r') as f:
    data = csv.reader(f, csv.QUOTE_NONE)
    print(list(data))

r = LogisticRegression()

