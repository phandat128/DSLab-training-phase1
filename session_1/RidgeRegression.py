import numpy as np
from numpy import ndarray
from pkgutil import get_data


def normalize_and_add_one(_x: ndarray) -> ndarray:
    x_max = np.array([[np.amax(_x[:, column_id])
                       for column_id in range(_x.shape[1])]
                      for _ in range(_x.shape[0])])
    x_min = np.array([[np.amin(_x[:, column_id])
                       for column_id in range(_x.shape[1])]
                      for _ in range(_x.shape[0])])
    x_normalized = (_x - x_min) / (x_max - x_min)
    one_column = np.array([[1] for _ in range(_x.shape[0])])

    return np.column_stack((one_column, x_normalized))


class RidgeRegression:
    def __init__(self):
        return

    def fit(self, _x_train: ndarray, _y_train: ndarray, _lambda: float) -> ndarray:
        # compute the matrix w with ridge session_1:
        # w = (XT.X + lambda.I).inv.X.Y
        return np.linalg.inv(_x_train.transpose().dot(_x_train) + _lambda * np.identity(_x_train.shape[1])) \
            .dot(_x_train.transpose()).dot(_y_train)

    def predict(self, _w: ndarray, _x_new: ndarray) -> ndarray:
        # compute the predicted value
        # y_predict = x_new * w
        return _x_new.dot(_w)

    def compute_rss(self, _y_new, _y_predicted):
        # rss = (1/n).sum(_y_new - y_predicted) ** 2)
        loss = 1. / (_y_new.shape[0]) * np.sum((_y_new - _y_predicted) ** 2)
        return loss

    def fit_gradient_descent(self, _x_train: ndarray, _y_train: ndarray,
                             _lambda: float, _learning_rate: float, _max_num_epoch=100, _batch_size=128):
        # loop to achieve the minimum loss
        _w = np.random.randn(_x_train.shape[1])  # initialize random w
        _last_loss = 10e8
        for ep in range(_max_num_epoch):
            _arr = np.array(range(_x_train.shape[0]))  # _arr = [0, 1, 2,..., N-1]
            np.random.shuffle(_arr)  # random shuffle the _arr array
            _x_train = _x_train[_arr]  # shuffle _x_train according to _arr
            _y_train = _y_train[_arr]  # shuffle _y_train according to _arr
            _total_minibatch = int(np.ceil(_x_train.shape[0] / _batch_size))
            for i in range(_total_minibatch):  # separate train data to some minibatch
                _index = i * _batch_size
                _x_train_sub: ndarray = _x_train[_index: _index + _batch_size]
                _y_train_sub: ndarray = _y_train[_index: _index + _batch_size]
                _grad = _x_train_sub.T.dot(_x_train_sub.dot(_w) - _y_train_sub) \
                        + _lambda * _w  # grad = XT.(Xw - y) + lamda.w
                _w = _w - _learning_rate * _grad  # update w
            _new_loss = self.compute_rss(_y_train, self.predict(_w, _x_train))
            if np.abs(_new_loss - _last_loss) < 1e-5:
                break
            _last_loss = _new_loss  # update new loss
        return _w

    def get_the_best_lambda(self, _x_train, _y_train):
        # do cross validation with potential lambda
        def cross_validation(_num_folds, _lambda):
            _row_ids = np.array(range(_x_train.shape[0]))
            # split row_ids to _num_folds part, test_ids[i] is used for test i
            _test_ids = np.split(_row_ids[:len(_row_ids) - len(_row_ids) % _num_folds],
                                 _num_folds)
            _test_ids[-1] = np.append(_test_ids[-1], _row_ids[len(_row_ids) - len(_row_ids) % _num_folds:])
            _train_ids = [[k for k in _row_ids if k not in _test_ids[i]]
                          for i in range(_num_folds)]  # train_ids[i] does not contain test_ids[i], is used for train i
            _sum_rss = 0
            for i in range(_num_folds):
                _test_parts = {'X': _x_train[_test_ids[i]], 'Y': _y_train[_test_ids[i]]}
                _train_parts = {'X': _x_train[_train_ids[i]], 'Y': _y_train[_train_ids[i]]}
                _w = self.fit(_train_parts['X'], _train_parts['Y'], _lambda)
                _y_predicted = self.predict(_w, _test_parts['X'])
                _sum_rss += self.compute_rss(_test_parts['Y'], _y_predicted)
            return _sum_rss / _num_folds

        # choose the best lambda with the minimun rss in lambda_values
        def range_scan(_best_lambda: float, _minimum_rss: float, _lambda_values):
            for _current_lambda in _lambda_values:
                average_rss = cross_validation(5, _current_lambda)
                if average_rss < _minimum_rss:
                    _best_lambda = _current_lambda
                    _minimum_rss = average_rss
                return _best_lambda, _minimum_rss

        _best_lambda, _minimum_rss = range_scan(0, 1000000, range(50))  # find best_lambda int
        print(_best_lambda)
        _lambda_values = [k * 1. / 1000 for k in range(
            max(0, (_best_lambda - 1) * 1000), (_best_lambda + 1) * 1000
        )]  # range from (best_lambda - 1) or 0 to (best_lambda + 1) with step 0.001
        _best_lambda, _minimum_rss = range_scan(_best_lambda, _minimum_rss, _lambda_values)
        return _best_lambda


def get_data(path):
    with open(path) as f:
        _data = f.readlines()[72:]
        _temp = []
        for _line in _data:
            _line = _line.strip()
            _row = [float(i) for i in (_line.split())]
            _temp.append(_row)
    _res = np.array(_temp)
    return _res[:, 1:-1], _res[:, -1]


if __name__ == '__main__':
    x, y = get_data("../datasets/x28.txt")
    x = normalize_and_add_one(x)
    x_train, y_train = x[:50], y[:50]
    x_test, y_test = x[50:], y[50:]

    ridge_regression = RidgeRegression()
    best_lambda = ridge_regression.get_the_best_lambda(x_train, y_train)
    print("Best LAMBDA: ", best_lambda)
    w_learned = ridge_regression.fit(x_train, y_train, best_lambda)
    y_predicted = ridge_regression.predict(w_learned, x_test)
    print("RSS: ", ridge_regression.compute_rss(y_test, y_predicted))
