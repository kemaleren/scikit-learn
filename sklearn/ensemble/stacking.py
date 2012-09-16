from .base import BaseEnsemble
import numpy as np
from itertools import izip
from ..grid_search import IterGrid
from ..base import ClassifierMixin, RegressorMixin
from ..utils.validation import assert_all_finite

__all__ = [
    "Stacking",
    "StackingC",
    'estimator_grid'
    ]


def estimator_grid(*args):
    result = []
    pairs = izip(args[::2], args[1::2])
    for estimator, params in pairs:
        for params in IterGrid(params):
            result.append(estimator(**params))
    return result


class MRLR(ClassifierMixin):
    """
    Converts a multi-class classification task into a set of
    indicator regression tasks.

    Ting, K.M., Witten, I.H.: Issues in stacked generalization. Journal of Artificial
    Intelligence Research 10, 271–289 (1999)

    """
    def __init__(self, regressor, **kwargs):
        self.estimator_ = regressor
        self.estimator_args = kwargs


    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        self.estimators_ = []
        for i in range(self.n_classes_):
            e = self.estimator_(**self.estimator_args)
            y_i = np.array(list(j == i for j in y))
            e.fit(X, y_i)
            self.estimators_.append(e)


    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


    def predict_proba(self, X):
        proba = []

        for i in range(self.n_classes_):
            e = self.estimators_[i]
            pred = e.predict(X).reshape(-1, 1)
            proba.append(pred)
        proba = np.hstack(proba)

        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba /= normalizer

        assert_all_finite(proba)
        assert np.all(proba.sum(axis=1) == 1)

        return proba



class Stacking(BaseEnsemble):
    """
    Implements stacking.

    David H. Wolpert (1992). Stacked generalization. Neural Networks,
    5:241-259, Pergamon Press.

    Params
    ------

    + meta_estimator : may be one of "best", "vote", "average", or any
      classifier or regressor constructor.

    + estimators : an iterable of estimators; each must support
      predict_proba()

    + cv : a cross validation object.

    + kwargs : arguments passed to instantiate meta_estimator.

    """

    # TODO: support different features for each estimator

    def __init__(self, meta_estimator, estimators, cv, **kwargs):
        self.estimators_ = estimators
        self.n_estimators_ = len(estimators)
        self.cv_ = cv

        if isinstance(meta_estimator, str):
            if meta_estimator not in ('best',
                                      'average',
                                      'vote'):
                raise Exception('invalid meta estimator: {0}'.format(meta_estimator))
            raise Exception('"{0}" meta estimator not implemented'.format(meta_estimator))
        elif issubclass(meta_estimator, ClassifierMixin):
            self.meta_estimator_ = meta_estimator(**kwargs)
        elif issubclass(meta_estimator, RegressorMixin):
            self.meta_estimator_ = MRLR(meta_estimator, **kwargs)
        else:
            raise Exception('invalid meta estimator: {0}'.format(meta_estimator))


    def _make_meta(self, X):
        rows = []
        for e in self.estimators_:
            proba = e.predict_proba(X)
            assert_all_finite(proba)
            rows.append(proba)
        return np.hstack(rows)


    def fit(self, X, y):
        # Build meta data #
        X_meta = []
        y_meta = []

        for a, b in self.cv_:
            X_a, X_b = X[a], X[b]
            y_a, y_b = y[a], y[b]

            for e in self.estimators_:
                e.fit(X_a, y_a)

            proba = self._make_meta(X_b)
            X_meta.append(proba)
            y_meta.append(y_b)

        X_meta = np.vstack(X_meta)
        if y_meta[0].ndim == 1:
            y_meta = np.hstack(y_meta)
        else:
            y_meta = np.vstack(y_meta)

        # train meta estimator #
        self.meta_estimator_.fit(X_meta, y_meta)

        # re-train estimators on full data #
        for e in self.estimators_:
            e.fit(X, y)


    def predict(self, X):
        X_meta = self._make_meta(X)
        return self.meta_estimator_.predict(X_meta)


    def predict_proba(self, X):
        X_meta = self._make_meta(X)
        return self.meta_estimator_.predict_proba(X_meta)


class StackingC(Stacking):
    """
    Implements StackingC.

    Seewald A.K.: How to Make Stacking Better and Faster While Also
    Taking Care of an Unknown Weakness, in Sammut C., Hoffmann A.
    (eds.), Proceedings of the Nineteenth International Conference on
    Machine Learning (ICML 2002), Morgan Kaufmann Publishers,
    pp.554-561, 2002.

    """
    pass


class FeatureWeightedLinearStacking(Stacking):
    """
    Implements Feature-Weighted Linear Stacking.

    Sill, J. and Takacs, G. and Mackey, L. and Lin, D.:
    Feature-weighted linear stacking. Arxiv preprint. 2009.

    """
    pass
