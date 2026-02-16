"""Model для MVC приложения"""
import numpy as np
from PySide6.QtCore import QObject, Signal
from funcs import OptimizationFunctions
from algs import OptimizationAlgorithms


class OptimizationModel(QObject):
    """Модель данных приложения"""
    
    # Сигналы для уведомления View об изменениях
    function_changed = Signal()
    algorithm_changed = Signal()
    optimization_step = Signal(list)  # Сигнал для каждого шага анимации
    optimization_finished = Signal()  # Сигнал окончания оптимизации
    
    def __init__(self):
        super().__init__()
        
        # Функция оптимизации
        self.funcs = OptimizationFunctions()
        self.current_function = 'sphere'
        self.function_params = {}
        
        # Алгоритм оптимизации
        self.algos = OptimizationAlgorithms()
        self.current_algorithm = 'gradient_descent'
        self.algorithm_params = {}
        
        # Путь оптимизации и генератор
        self.optimization_path = []
        self.optimization_generator = None
        
        # Диапазон для построения графика
        self.x_range = (-5, 5)
        self.y_range = (-5, 5)
        self.grid_size = 100
        
    def set_function(self, func_name, params=None):
        """Устанавливает текущую функцию"""
        self.current_function = func_name
        if params:
            self.function_params = params
        else:
            # Устанавливаем параметры по умолчанию
            param_defs = self.funcs.get_function_params(func_name)
            self.function_params = {p['name']: p['default'] for p in param_defs}
        
        self.function_changed.emit()
        
    def set_algorithm(self, algo_name, params=None):
        """Устанавливает текущий алгоритм"""
        self.current_algorithm = algo_name
        if params:
            self.algorithm_params = params
        else:
            # Устанавливаем параметры по умолчанию
            param_defs = self.algos.get_algorithm_params(algo_name)
            self.algorithm_params = {p['name']: p['default'] for p in param_defs}
        
        self.algorithm_changed.emit()
        
    def update_function_param(self, param_name, value):
        """Обновляет параметр функции"""
        self.function_params[param_name] = value
        self.function_changed.emit()
        
    def update_algorithm_param(self, param_name, value):
        """Обновляет параметр алгоритма"""
        self.algorithm_params[param_name] = value
        self.algorithm_changed.emit()
        
    def get_function_data(self):
        """Возвращает данные функции для построения графика"""
        x = np.linspace(self.x_range[0], self.x_range[1], self.grid_size)
        y = np.linspace(self.y_range[0], self.y_range[1], self.grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Получаем функцию
        func = getattr(self.funcs, self.current_function)
        Z = func(X, Y, **self.function_params)
        
        return X, Y, Z
    
    def start_optimization(self, x0, y0):
        """Запускает алгоритм оптимизации - создает генератор"""
        func = getattr(self.funcs, self.current_function)
        algo = getattr(self.algos, self.current_algorithm)
        
        # Сбрасываем путь
        self.optimization_path = []
        
        # Создаем генератор
        self.optimization_generator = algo(
            func, x0, y0, 
            **self.algorithm_params,
            **self.function_params
        )
        
    def step_optimization(self):
        """Выполняет один шаг оптимизации"""
        if self.optimization_generator is None:
            return False
            
        try:
            point = next(self.optimization_generator)
            self.optimization_path.append(point)

            self.optimization_step.emit(self.optimization_path.copy())
            return True
            
        except StopIteration:
            self.optimization_generator = None
            self.optimization_finished.emit()
            return False
    
    def get_available_functions(self):
        """Возвращает список доступных функций"""
        return self.funcs.get_available_functions()
    
    def get_available_algorithms(self):
        """Возвращает список доступных алгоритмов"""
        return self.algos.get_available_algorithms()
    
    def get_function_params_info(self, func_name):
        """Возвращает информацию о параметрах функции"""
        return self.funcs.get_function_params(func_name)
    
    def get_algorithm_params_info(self, algo_name):
        """Возвращает информацию о параметрах алгоритма"""
        return self.algos.get_algorithm_params(algo_name)