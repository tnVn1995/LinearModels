import numpy as np
from numpy import array
import fractions
from typing import List

np.set_printoptions(formatter= \
                        {'all': lambda x: str(fractions.Fraction(x).limit_denominator())})


class LinearModels(object):
    def __init__(self):
        pass

    # noinspection PyAttributeOutsideInit
    def projection(self, x: array, y: array) -> array:
        """Return the projection of x onto y
        Input
        -----
        x,y: [array-like] shape: [n, 1]
        Output
        ------
        return the projection of x onto y"""
        projection_ = (x.T.dot(y) / np.sum(y ** 2)) * y
        # print('The projection is:\n', self.projection_)
        return projection_

    def spaceprojection(self, x: array, Y: List[array]) -> array:
        """Find the projection of a vector onto a vector space"""
        projections = np.array([self.projection(x, y) for y in Y])
        return np.sum(projections)

    def gramschmidt(self, X: List[array]) -> List[array]:
        """Perform GS orthogonalization process"""
        for idx, val in enumerate(X):

        pass
