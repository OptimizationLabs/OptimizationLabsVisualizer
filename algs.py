"""Алгоритмы поиска минимума"""
import numpy as np


class OptimizationAlgorithms:
    """Класс с алгоритмами оптимизации"""
    
    @staticmethod
    def gradient_descent(func, x0, y0, learning_rate=0.01, iterations=100, **func_params):
        """Градиентный спуск"""
        x, y = x0, y0
        h = 1e-5
        
        # Начальная точка
        yield (x, y)
        
        for _ in range(iterations):
            # Численный градиент
            grad_x = (func(x + h, y, **func_params) - func(x - h, y, **func_params)) / (2 * h)
            grad_y = (func(x, y + h, **func_params) - func(x, y - h, **func_params)) / (2 * h)
            
            # Обновление координат
            x = x - learning_rate * grad_x
            y = y - learning_rate * grad_y
            
            yield (x, y)

    @staticmethod
    def get_algorithm_params(algo_name):
        """Возвращает параметры алгоритма"""
        params = {
            'gradient_descent': [
                {'name': 'learning_rate', 'default': 0.01, 'min': 0.001, 'max': 0.1, 'step': 0.001},
                {'name': 'iterations', 'default': 100, 'min': 10, 'max': 500, 'step': 10}
            ],
        }
        return params.get(algo_name, [])
    
    @staticmethod
    def get_available_algorithms():
        """Возвращает список доступных алгоритмов"""
        return ['gradient_descent']