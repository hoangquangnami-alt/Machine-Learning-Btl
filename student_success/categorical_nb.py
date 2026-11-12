"""Fold-local categorical representation and support-safe CategoricalNB."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer, OrdinalEncoder
from sklearn.utils.validation import check_is_fitted
from .data import CATEGORICAL, NUMERIC, SEED


def shift_categories(values):
    """Known values become 1..K; the encoder's -1 unknown becomes 0."""
    return values.astype(np.int64) + 1


class SupportedCategoricalNB(ClassifierMixin, BaseEstimator):
    """Allocate all observed codes plus the reserved zero code for each field.

    Counts are determined solely from the transformed training fold. Category
    encodings are contiguous and unknown is zero, including when absent in fit.
    """
    def __init__(self, alpha=1.0, fit_prior=True):
        self.alpha = alpha
        self.fit_prior = fit_prior

    def fit(self, X, y):
        X = np.asarray(X)
        if not np.all(np.isfinite(X)) or np.any(X < 0) or not np.all(X == np.floor(X)):
            raise ValueError('CategoricalNB requires nonnegative integer codes')
        self.min_categories_ = np.maximum(X.max(axis=0).astype(int) + 1, 1)
        self.model_ = CategoricalNB(alpha=self.alpha, fit_prior=self.fit_prior,
                                    min_categories=self.min_categories_)
        self.model_.fit(X.astype(np.int64), y)
        self.classes_ = self.model_.classes_
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, 'model_')
        return self.model_.predict(np.asarray(X, dtype=np.int64))

    def predict_proba(self, X):
        check_is_fitted(self, 'model_')
        return self.model_.predict_proba(np.asarray(X, dtype=np.int64))


def build_pipeline():
    cats = Pipeline([('impute', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
                     ('encode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
                     ('reserve_unknown', FunctionTransformer(shift_categories))])
    nums = Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                     ('bins', KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile',
                                               quantile_method='averaged_inverted_cdf', dtype=np.float64,
                                               subsample=None, random_state=SEED))])
    return Pipeline([('preprocess', ColumnTransformer([('categorical', cats, CATEGORICAL),
                                                       ('numeric', nums, NUMERIC)])),
                     ('model', SupportedCategoricalNB())])
