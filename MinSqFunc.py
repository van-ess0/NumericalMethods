import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from math import fabs

eps = 0.0000001


def function(point):
    (x, y, z) = point
    return 2 * x ** 2 + (3 + 0.1 * 9) * y ** 2 + (4 + 0.1 * 9) * z ** 2 + x * y - y * z + x * z + x - 2 * y + 3 * z + 9


a = tangent.grad(function)

def grad_fun(point):
    print(point)
    ans1 = lambda point: 4 * point[0] + point[1] + point[2] + 1
    ans2 = lambda point: 2 * (3 + 0.9) * point[1] + point[0] - point[2] - 2
    ans3 = lambda point: 2 * (4 + 0.9) * point[2] - point[1] + point[0] + 3
    return np.array([ans1(point), ans2(point), ans3(point)])

def grad(func, point):
    # res = np.array([der(func, i, point) for i in range(1, 4)])
    return res


def der(func, arg, point):
    """
    Derivative of function in some point by one of its args
    :param func: Function (function or lambda)
    :param arg: Argument of function
    :param point: Where to search derivative
    """
    increment = np.zeros(3)
    increment[arg - 1] += eps / 10
    # FIXME!!!
    ans = (func(np.array(point) + increment) - func(np.array(point) - increment)) / (eps / 5)

    return ans


def one_dimension_optimization(func, a, b, eps):
    """

    :param func:
    :param a:
    :param b:
    :param eps:
    :return:
    """
    c = (b + a) / 2
    # print(a, b, c)
    if fabs(b - a) < eps:
        return c
    if func(c + eps) < func(c - eps):
        return one_dimension_optimization(func, c, b, eps)
    else:
        return one_dimension_optimization(func, a, c, eps)


def fastest_gradient_descent(func, start=None):
    if start is None:
        start = np.array([-10465, -1065, 10000])
    gr = grad_fun(start)
    # print(gr)
    a = one_dimension_optimization(
        lambda a: func(start - gr.dot(a)),
        0., 100.,
        eps / 100,
    )
    # a = minimize(lambda a: func(start - gr.dot(a)), 0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
    end = start - gr.dot(a)
    # print(end)
    print('norm:', np.linalg.norm(start - end))
    return end if (np.linalg.norm(start - end) < eps) else fastest_gradient_descent(func, end)

def fastest_coordinate_descent(func, start=None, coord=0):
    if start is None:
        start = np.array([-10465, -1065, 10000])
    gr = np.eye(3)[coord]
    # print(gr)
    a = one_dimension_optimization(
        lambda a: func(start - gr.dot(a)),
        -50., 50.,
        eps / 100,
    )
    # a = minimize(lambda a: func(start - gr.dot(a)), 0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
    end = start - gr.dot(a)
    # print(end)
    print('norm:', np.linalg.norm(start - end))
    return end if (np.linalg.norm(start - end) < eps) else fastest_coordinate_descent(func, end, (coord + 1) % 3)


print(fastest_coordinate_descent(function))
