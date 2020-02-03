import numpy as np
from numpy import array
import fractions

np.set_printoptions(formatter=\
                        {'all': lambda x: str(fractions.Fraction(x).limit_denominator())})


class LinearModels(object):
    def __init__(self):
        pass

    def projection(self, x: array, y: array) -> array:
        """Return the projection of x onto y
        Input
        -----
        x,y: [array-like]
        Output
        ------
        return the projection of x onto y"""
        self.projection_ = (x.T.dot(y) / np.sum(y ** 2)) * y
        # print('The projection is:\n', self.projection_)
        return self.projection_

    def GramSchmidt(self):
        """Perform GS orthogonalization process"""
        pass