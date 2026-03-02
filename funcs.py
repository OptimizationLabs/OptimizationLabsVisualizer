import numpy as np


class OptimizationFunctions:
    """Класс с функциями для оптимизации"""
    
    @staticmethod
    def sphere(x, y, a=1.0):
        return a * (x**2 + y**2)

    @staticmethod
    def himmelblau(x, y):
        return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

    @staticmethod
    def rastrigin(x, y, a=10.0):
        """Функция Растригина. Глобальный минимум: f(0, 0) = 0"""
        return 2 * a + (x ** 2 - a * np.cos(2 * np.pi * x)) + \
                       (y ** 2 - a * np.cos(2 * np.pi * y))

    @staticmethod
    def get_function_params(func_name):
        """Возвращает параметры функции"""
        params = {
            'sphere': [
                {'name': 'a', 'default': 1.0, 'min': 0.1, 'max': 10.0, 'step': 0.1}
            ],
            'himmelblau': [],
            'rastrigin': [
                {'name': 'a', 'default': 10.0, 'min': 0.1, 'max': 20.0, 'step': 0.1}
            ],
        }
        return params.get(func_name, [])

    @staticmethod
    def get_available_functions():
        funcs = [
            name
            for name, value in OptimizationFunctions.__dict__.items()
            if isinstance(value, staticmethod) and (not name.startswith('_'))
        ]
        funcs.remove('get_function_params')
        funcs.remove('get_available_functions')
        return funcs