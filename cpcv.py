import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import matplotlib.patches as patches
import matplotlib.dates as mdates
import itertools as itt
import numbers
import numpy as np
import pandas as pd

from abc import abstractmethod
from typing import Iterable, Tuple, List
from scipy.optimize import brute
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import KFold
from itertools import combinations

class BaseTimeSeriesCrossValidator:
    """
    Abstract class for time series cross-validation.

    Time series cross-validation requires each sample has a prediction time pred_time, at which the features are used to
    predict the response, and an evaluation time eval_time, at which the response is known and the error can be
    computed. Importantly, it means that unlike in standard sklearn cross-validation, the samples X, response y,
    pred_times and eval_times must all be pandas dataframe/series having the same index. It is also assumed that the
    samples are time-ordered with respect to the prediction time (i.e. pred_times is non-decreasing).

    Parameters
    ----------
    n_splits : int, default=10
        Number of folds. Must be at least 2.

    """
    def __init__(self, n_splits=10):
        if not isinstance(n_splits, numbers.Integral):
            raise ValueError(f"The number of folds must be of Integral type. {n_splits} of type {type(n_splits)}"
                             f" was passed.")
        n_splits = int(n_splits)
        if n_splits <= 1:
            raise ValueError(f"K-fold cross-validation requires at least one train/test split by setting n_splits = 2 "
                             f"or more, got n_splits = {n_splits}.")
        self.n_splits = n_splits
        self.pred_times = None
        self.eval_times = None
        self.indices = None

    @abstractmethod
    def split(self, X: pd.DataFrame, y: pd.Series = None,
              pred_times: pd.Series = None, eval_times: pd.Series = None):
        if not isinstance(X, pd.DataFrame) and not isinstance(X, pd.Series):
            raise ValueError('X should be a pandas DataFrame/Series.')
        if not isinstance(y, pd.Series) and y is not None:
            raise ValueError('y should be a pandas Series.')
        if not isinstance(pred_times, pd.Series):
            raise ValueError('pred_times should be a pandas Series.')
        if not isinstance(eval_times, pd.Series):
            raise ValueError('eval_times should be a pandas Series.')
        if y is not None and (X.index == y.index).sum() != len(y):
            raise ValueError('X and y must have the same index')
        if (X.index == pred_times.index).sum() != len(pred_times):
            raise ValueError('X and pred_times must have the same index')
        if (X.index == eval_times.index).sum() != len(eval_times):
            raise ValueError('X and eval_times must have the same index')

        if not pred_times.equals(pred_times.sort_values()):
            raise ValueError('pred_times should be sorted')
        if not eval_times.equals(eval_times.sort_values()):
            raise ValueError('eval_times should be sorted')

        self.pred_times = pred_times
        self.eval_times = eval_times
        self.indices = np.arange(X.shape[0])

def back_test_paths_generator(t_span, n, k, prediction_times, evaluation_times, verbose=True):
    # split data into N groups, with N << T
    # this will assign each index position to a group position
    group_num = np.arange(t_span) // (t_span // n)
    group_num[group_num == n] = n-1
    
    # generate the combinations 
    test_groups = np.array(list(itt.combinations(np.arange(n), k))).reshape(-1, k)
    C_nk = len(test_groups)
    n_paths = C_nk * k // n 
    
    
    if verbose:
        print('n_sim:', C_nk)
        print('n_paths:', n_paths)
    
    # is_test is a T x C(n, k) array where each column is a logical array 
    # indicating which observation in in the test set
    is_test_group = np.full((n, C_nk), fill_value=False)
    is_test = np.full((t_span, C_nk), fill_value=False)
    
    # assign test folds for each of the C(n, k) simulations
    for k, pair in enumerate(test_groups):
        i, j = pair
        is_test_group[[i, j], k] = True
        
        # assigning the test folds
        mask = (group_num == i) | (group_num == j)
        is_test[mask, k] = True
        
    # for each path, connect the folds from different simulations to form a backtest path
    # the fold coordinates are: the fold number, and the simulation index e.g. simulation 0, fold 0 etc
    path_folds = np.full((n, n_paths), fill_value=np.nan)
    
    for i in range(n_paths):
        for j in range(n):
            s_idx = is_test_group[j, :].argmax().astype(int)
            path_folds[j, i] = s_idx
            is_test_group[j, s_idx] = False
            cv.split(X, y, pred_times=prediction_times, eval_times=evaluation_times)
    
    # finally, for each path we indicate which simulation we're building the path from and the time indices
    paths = np.full((t_span, n_paths), fill_value= np.nan)
    
    for p in range(n_paths):
        for i in range(n):
            mask = (group_num == i)
            paths[mask, p] = int(path_folds[i, p])
    # paths = paths_# .astype(int)

    return (is_test, paths, path_folds)   

def compute_fold_bounds(cv: BaseTimeSeriesCrossValidator, split_by_time: bool) -> List[int]:
    """
    Compute a list containing the fold (left) boundaries.

    Parameters
    ----------
    cv: BaseTimeSeriesCrossValidator
        Cross-validation object for which the bounds need to be computed.
    split_by_time: bool
        If False, the folds contain an (approximately) equal number of samples. If True, the folds span identical
        time intervals.
    """
    if split_by_time:
        full_time_span = cv.pred_times.max() - cv.pred_times.min()
        fold_time_span = full_time_span / cv.n_splits
        fold_bounds_times = [cv.pred_times.iloc[0] + fold_time_span * n for n in range(cv.n_splits)]
        return cv.pred_times.searchsorted(fold_bounds_times)
    else:
        return [fold[0] for fold in np.array_split(cv.indices, cv.n_splits)]


def embargo(cv: BaseTimeSeriesCrossValidator, train_indices: np.ndarray,
            test_indices: np.ndarray, test_fold_end: int) -> np.ndarray:
    """
    Apply the embargo procedure to part of the train set.

    This amounts to dropping the train set samples whose prediction time occurs within self.embargo_dt of the test
    set sample evaluation times. This method applies the embargo only to the part of the training set immediately
    following the end of the test set determined by test_fold_end.

    Parameters
    ----------
    cv: Cross-validation class
        Needs to have the attributes cv.pred_times, cv.eval_times, cv.embargo_dt and cv.indices.

    train_indices: np.ndarray
        A numpy array containing all the indices of the samples currently included in the train set.

    test_indices : np.ndarray
        A numpy array containing all the indices of the samples in the test set.

    test_fold_end : int
        Index corresponding to the end of a test set block.

    Returns
    -------
    train_indices: np.ndarray
        The same array, with the indices subject to embargo removed.

    """
    if not hasattr(cv, 'embargo_td'):
        raise ValueError("The passed cross-validation object should have a member cv.embargo_td defining the embargo"
                         "time.")
    last_test_eval_time = cv.eval_times.iloc[test_indices[test_indices <= test_fold_end]].max()
    min_train_index = len(cv.pred_times[cv.pred_times <= last_test_eval_time + cv.embargo_td])
    if min_train_index < cv.indices.shape[0]:
        allowed_indices = np.concatenate((cv.indices[:test_fold_end], cv.indices[min_train_index:]))
        train_indices = np.intersect1d(train_indices, allowed_indices)
    return train_indices


def purge(cv: BaseTimeSeriesCrossValidator, train_indices: np.ndarray,
          test_fold_start: int, test_fold_end: int) -> np.ndarray:
    """
    Purge part of the train set.

    Given a left boundary index test_fold_start of the test set, this method removes from the train set all the
    samples whose evaluation time is posterior to the prediction time of the first test sample after the boundary.

    Parameters
    ----------
    cv: Cross-validation class
        Needs to have the attributes cv.pred_times, cv.eval_times and cv.indices.

    train_indices: np.ndarray
        A numpy array containing all the indices of the samples currently included in the train set.

    test_fold_start : int
        Index corresponding to the start of a test set block.

    test_fold_end : int
        Index corresponding to the end of the same test set block.

    Returns
    -------
    train_indices: np.ndarray
        A numpy array containing the train indices purged at test_fold_start.

    """
    time_test_fold_start = cv.pred_times.iloc[test_fold_start]
    # The train indices before the start of the test fold, purged.
    train_indices_1 = np.intersect1d(train_indices, cv.indices[cv.eval_times < time_test_fold_start])
    # The train indices after the end of the test fold.
    train_indices_2 = np.intersect1d(train_indices, cv.indices[test_fold_end:])
    return np.concatenate((train_indices_1, train_indices_2))