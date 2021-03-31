''' this file contains all sort of numeric methods'''

#THIS FUNCTION IS BASED ON THE BEZIER CURVE METHOD
def interpolate(f: callable, a: float, b: float, n: int) -> callable:
    """
    Interpolate the function f in the closed range [a,b] using at most n
    points. Your main objective is minimizing the interpolation error.
    Your secondary objective is minimizing the running time.
    The assignment will be tested on variety of different functions with
    large n values.

    Interpolation error will be measured as the average absolute error at
    2*n random points between a and b. See test_with_poly() below.

    Note: It is forbidden to call f more than n times.

    Note: This assignment can be solved trivially with running time O(n^2)
    or it can be solved with running time of O(n) with some preprocessing.
    **Accurate O(n) solutions will receive higher grades.**

    Note: sometimes you can get very accurate solutions with only few points,
    significantly less than n.

    Parameters
    ----------
    f : callable. it is the given function
    a : float
        beginning of the interpolation range.
    b : float
        end of the interpolation range.
    n : int
        maximal number of points to use.

    Returns
    -------
    The interpolating function --> for each x value it will return the matching y value.

    """

    def b_i_poly(i, n, t):
        """
         The B_i polynomial of n, i as a function of t
        """

        x = comb(n, i)
        return x * (t ** (n - i)) * (1 - t) ** i

    def bezier_curve(points, nTimes=1000):
        """
           Given a list of points we chose by n equal spaces , return the
           bezier curve defined by those points.
           nTimes is the number of time steps, defaults to 1000
        """

        nPoints = len(points)
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])

        t = np.linspace(0.0, 1.0, nTimes)

        polynomial_array = np.array([b_i_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

        xvals = np.dot(xPoints, polynomial_array)
        yvals = np.dot(yPoints, polynomial_array)

        return xvals, yvals

    if n == 1:
        po = b - a / 2
        points = [(a, f(a)), (po, f(po)), (b, f(b))]
        xvals, yvals = bezier_curve(points, nTimes=1000)
    else:
        po = a
        c = b - a / n
        points = [(a, f(a))]
        while len(points) < n - 1:
            po += c
            points.append((po, f(po)))
        points.append((b, f(b)))
        xvals, yvals = bezier_curve(points, nTimes=1000)

    def createfun(x, ypoints, xpoints):
        for i in range(len(xpoints) - 1):
            if x <= xpoints[i] and x >= xpoints[i + 1]:
                return ypoints[i + 1]

    return lambda x: createfun(x, yvals, xvals)

#THIS function is based on the Newton-Raphson algorithm is a commonly used technique for locating zeros of a function
def intersections(f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
    """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.

        This function may not work correctly if there is infinite number of
        intersection points.


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """
    result = f1 - f2
    if result == 0:
        return np.math.inf
    result_der = np.polyder(result, m=1)
    n = len(result)
    inter_lst = []
    x = np.random.uniform(a, b)
    h = result(x) / result_der(x)
    for i in range(n):
        x = np.random.uniform(a, b)
        count = 0
        while abs(h) >= maxerr and count < 100:
            x = x - (result(x) / result_der(x))
            count += 1

        if count > 100:
            continue
        if x not in inter_lst and a <= x <= b:
            inter_lst.append(x)

    return inter_lst

#THIS function is based on the simpson's rule
def integrate(f: callable, a: float, b: float, n: int) -> np.float32:
    """
        Integrate the function f in the closed range [a,b] using at most n
        points. Your main objective is minimizing the integration error.
        Your secondary objective is minimizing the running time. The assignment
        will be tested on variety of different functions.

        Integration error will be measured compared to the actual value of the
        definite integral.

        Note: It is forbidden to call f more than n times.

        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the integration range.
        b : float
            end of the integration range.
        n : int
            maximal number of points to use.

        Returns
        -------

            The definite integral of f between a and b
        """
    h = (b - a) / float(n)
    result = f(a) + f(b)

    for i in range(1, n, 1):
        if i % 2 == 0:
            result += (2 * (f(a + i * h)))
        else:
            result += (4 * (f(a + i * h)))
    result *= (h / 3.0)
    return np.float32(result)
