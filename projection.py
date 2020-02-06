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
        projection_ = (x.T.dot(y) / (y.T.dot(y))) * y
        # print('The projection is:\n', self.projection_)
        return projection_

    def spaceprojection(self, x: array, Y: List[array]) -> array:
        """Find the projection of a vector onto a vector space"""
        projections = np.array([self.projection(x, y) for y in Y])
        return np.sum(projections, axis=0)

    def gramschmidt(self, X: List[array], debug=False) -> List[array]:
        """Perform GS orthogonalization process"""
        v1 = X[0]
        bases = [v1]
        if debug:
            for idx, val in enumerate(X[1:]):
                bases.append(val - self.spaceprojection(val, bases))
                print(f'The bases after {idx + 1} run is:\t', bases)
        else:
            for idx, val in enumerate(X[1:]):
                bases.append(val - self.spaceprojection(val, bases))
        return bases
    def projection_matrix(self, X: array):
        """Return projection matrix of full rank matrix X"""
        return X.dot(np.linalg.inv(X.T.dot(X))).dot(X.T)
    # def __str__(self):
    #     return 'Utility functions for TTU Linear Model class'
    #
    # def __repr__(self):
    #     return 'Let\'s make some utility functions for TTU Linear Model'


##
if __name__ == '__main__':
    LinearModel = LinearModels()
    x1 = np.array([[1, 1, 1, 1]]).T
    x2 = np.array([[4, 1, 3, 4]]).T
    y = np.array([[1, 9, 5, 5]]).T
    projection1 = LinearModel.projection(x2, x1)
    assert np.array_equal(projection1, np.array([[3, 3, 3, 3]]).T), 'Your result is incorrect!'
    v2 = LinearModel.gramschmidt([x1, x2])
    assert np.array_equal(v2, [x1, np.array([[1, -2, 0, 1]]).T]), 'Your gram-schmidt orthog is incorrect!'

    # x1_test = np.array([[1,2,3,0]]).T
    # x2_test = np.array([[1,2,0,0]]).T
    # x3_test = np.array([[1,0,0,1]]).T
    # X_test = [x1_test,x2_test,x3_test]

## test

# import numpy as np
# X = np.array([[2,-1,-1]]).T
# X = np.array([[1,1,1],[1,-1,0]]).T
# X = np.array([[1,1,1,1],[1,-2,0,1]]).T
# P = X.dot(np.linalg.inv(X.T.dot(X))).dot(X.T)
# y = np.array([[1,9,5,5]]).T
# y.T.dot(P).dot(y)