import numpy as np


class OptimizationFunctions:
    """Класс с функциями для оптимизации"""
    
    @staticmethod
    def sphere(x, y, a=1.0):
        return a * (x**2 + y**2)

    @staticmethod
    def himmelblau(x, y):
        return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

    @staticmethod
    def rastrigin(x, y, a=10.0):
        return 2*a + (x**2 - a * np.cos(2 * np.pi * x)) + \
                     (y**2 - a * np.cos(2 * np.pi * y))
    
    @staticmethod
    def quadratic(x, y):
        """
        Функция из лабораторной работы №2
        f(x,y) = 2x^2 + 2xy + 2y^2 - 4x - 6y
        """
        return 2*x*x + 2*x*y + 2*y*y - 4*x - 6*y

    @staticmethod
    def schwefel(x, y, scale=500):
        """Функция Швефеля"""
        return -1 * (
            x * scale * np.sin(np.sqrt(np.abs(x * scale))) +
            y * scale * np.sin(np.sqrt(np.abs(y * scale)))
        ) / scale
        

    @staticmethod
    def get_function_params(func_name):
        params = {
            'sphere': [
                {'name': 'a', 'default': 1.0, 'min': 0.1, 'max': 10.0, 'step': 0.1}
            ],
            'himmelblau': [],
            'rastrigin': [
                {'name': 'a', 'default': 10.0, 'min': 0.1, 'max': 20.0, 'step': 0.1}
            ],
            'quadratic': [],
            'schwefel': []
        }
        return params.get(func_name, [])

    @staticmethod
    def get_available_functions():
        funcs = []
        for name, value in OptimizationFunctions.__dict__.items():
            if isinstance(value, staticmethod) and not name.startswith('_'):
                if name not in ['get_function_params', 'get_available_functions']:
                    funcs.append(name)
        return funcs
