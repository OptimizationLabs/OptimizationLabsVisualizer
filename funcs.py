"""Функции для оптимизации"""
import numpy as np


class OptimizationFunctions:
    """Класс с функциями для оптимизации"""
    
    @staticmethod
    def sphere(x, y, a=1.0):
        """Функция сферы: f(x,y) = a*(x^2 + y^2)"""
        return a * (x**2 + y**2)
    
    @staticmethod
    def get_function_params(func_name):
        """Возвращает параметры функции"""
        params = {
            'sphere': [
                {'name': 'a', 'default': 1.0, 'min': 0.1, 'max': 10.0, 'step': 0.1}
            ],
        }
        return params.get(func_name, [])
    
    @staticmethod
    def get_available_functions():
        """Возвращает список доступных функций"""
        return ['sphere']