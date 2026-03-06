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
    def lab2_function(x, y):
        """
        Функция из лабораторной работы №2
        f(x,y) = 2x^2 + 2xy + 2y^2 - 4x - 6y
        Минимум: f(1/3, 5/6) = -25/6 ≈ -4.1667
        """
        return 2*x*x + 2*x*y + 2*y*y - 4*x - 6*y

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
            'lab2_function': [],
        }
        return params.get(func_name, [])

    @staticmethod
    def get_available_functions():
        funcs = [
            name
            for name, value in OptimizationFunctions.__dict__.items()
            if isinstance(value, staticmethod) and (not name.startswith('_'))
        ]
        # Убираем служебные методы
        for method in ['get_function_params', 'get_available_functions']:
            if method in funcs:
                funcs.remove(method)
        return funcs