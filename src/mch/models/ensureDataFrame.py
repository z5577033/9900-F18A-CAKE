from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class EnsureDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._cols_ = None
        self._index_ = None

    def fit(self, X, y=None):
        # Capture column/index if X is already a DataFrame
        if isinstance(X, pd.DataFrame):
            self._cols_ = X.columns.astype(str).tolist()
            self._index_ = X.index.astype(str)
        else:
            # if array, defer to transform to build names
            self._cols_ = None
            self._index_ = None
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            df.columns = df.columns.astype(str)
            df.index = df.index.astype(str)
            return df
        # X is array-like; make columns/index if we don't have them
        X = np.asarray(X)
        n_rows, n_cols = X.shape
        cols = self._cols_ if self._cols_ is not None else [str(i) for i in range(n_cols)]
        idx  = self._index_ if self._index_ is not None else [f"s_{i}" for i in range(n_rows)]
        df = pd.DataFrame(X, columns=[str(c) for c in cols], index=[str(i) for i in idx])
        return df
