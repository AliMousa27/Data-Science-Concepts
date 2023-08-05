from vectors import Vector

#given m and c we can predict the output y from y=mx+c
def predict(m: float, c: float, x_i: float) -> float:
    return m * x_i + c

def error(m: float, c: float, x_i: float, y_i: float) -> float:
    """
    The error from predicting m * x_i + c
    when the actual value is y_i
    """
    return predict(m,c,x_i)-y_i

def sum_of_sqerrors(m: float, c: float, x: Vector, y: Vector) -> float:
    return [sum(error(m,c,xi,yi)**2) for xi,yi in zip(x,y)]