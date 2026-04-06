from math import sqrt

def gradient_descent(func, x0, y0, t, eps, eps1, eps2, M, **func_params):
    """
    Градиентный спуск с делением шага (backtracking)
    """
    iterations = int(M)
    k = 0
    h = 1e-5

    x, y = x0, y0
    yield ((x, y),)
    while k < iterations:
        grad_x = (func(x + h, y, **func_params) - func(x - h, y, **func_params)) / (2 * h)
        grad_y = (func(x, y + h, **func_params) - func(x, y - h, **func_params)) / (2 * h)

        grad_norm = sqrt(grad_x**2 + grad_y**2)

        if grad_norm <= eps1:
            return

        t_k = t
        f_current = func(x, y, **func_params)
        while True:
            tmp_x = x - t_k * grad_x
            tmp_y = y - t_k * grad_y
            f_next = func(tmp_x, tmp_y, **func_params)

            if f_next <= f_current - eps * t_k * grad_norm**2:
                break
            t_k /= 2

        step_norm = sqrt((tmp_x - x)**2 + (tmp_y - y)**2)
        func_diff = abs(f_next - f_current)

        x_new, y_new = tmp_x, tmp_y
        yield ((x_new, y_new),)

        if step_norm < eps2 and func_diff < eps2:
            return

        x, y = x_new, y_new
        k += 1
        