from math import sqrt


class OptimizationAlgorithms:
    """Класс с алгоритмами оптимизации"""

    @staticmethod
    def gradient_descent(func, x0, y0, t, eps, eps1, eps2, M, **func_params):
        """
        Градиентный спуск с делением шага (backtracking)
        """

        iterations = int(M)
        k = 0
        h = 1e-5

        x, y = x0, y0
        yield (x, y)
        while k < iterations:
            grad_x = (func(x + h, y, **func_params) - func(x - h, y, **func_params)) / (2 * h)
            grad_y = (func(x, y + h, **func_params) - func(x, y - h, **func_params)) / (2 * h)

            grad_norm = sqrt(grad_x ** 2 + grad_y ** 2)

            if grad_norm <= eps1:
                return (x, y)
            # Backtracking
            t_k = t
            f_current = func(x, y, **func_params)
            while True:
                tmp_x = x - t_k * grad_x
                tmp_y = y - t_k * grad_y

                f_next = func(tmp_x, tmp_y, **func_params)

                if f_next <= f_current - eps * t_k * grad_norm ** 2:
                    break
                else:
                    t_k /= 2

            step_norm = sqrt((tmp_x - x) ** 2 + (tmp_y - y) ** 2)
            func_diff = abs(f_next - f_current)

            # обновляем точку
            x_new, y_new = tmp_x, tmp_y

            yield (x_new, y_new)

            if step_norm < eps2 and func_diff < eps2:
                return (x_new, y_new)
            x, y = x_new, y_new
            k += 1

        return (x, y)

    @staticmethod
    def get_algorithm_params(algo_name):
        """Возвращает параметры алгоритма"""

        params = {
            'gradient_descent': [
                {
                    'name': 't',
                    'label': 'Initial step (t)',
                    'default': 0.01,
                    'min': 0.0001,
                    'max': 1.0,
                    'step': 0.0001,
                },
                {
                    'name': 'eps',
                    'label': 'coefficient (eps)',
                    'default': 0.1,
                    'min': 1e-12,
                    'max': 1e-1,
                    'step': 1e-12,
                },
                {
                    'name': 'eps1',
                    'label': 'Gradient tolerance (eps1)',
                    'default': 1e-4,
                    'min': 1e-12,
                    'max': 1e-1,
                    'step': 1e-12,
                },
                {
                    'name': 'eps2',
                    'label': 'Step tolerance (eps2)',
                    'default': 1e-6,
                    'min': 1e-12,
                    'max': 1e-1,
                    'step': 1e-12,
                },
                {
                    'name': 'M',
                    'label': 'Max iterations (M)',
                    'default': 200,
                    'type': int,
                    'min': 10,
                    'max': 2000,
                    'step': 10,
                },
            ],
        }
        return params.get(algo_name, [])

    @staticmethod
    def get_available_algorithms():
        """Возвращает список доступных алгоритмов"""
        return ['gradient_descent']