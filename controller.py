"""Controller для MVC приложения"""
from PySide6.QtCore import QObject, QTimer


class OptimizationController(QObject):
    """Контроллер приложения"""
    
    def __init__(self, model, view):
        super().__init__()
        self.model = model
        self.view = view
        
        # Таймер для анимации
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.on_animation_step)
        self.animation_speed = 50  # миллисекунд между шагами
        
        # Соединяем сигналы и слоты
        self.connect_signals()
        
        # Инициализация
        self.initialize()
        
    def connect_signals(self):
        """Соединяет сигналы из View и Model"""
        
        # Сигналы от Model
        self.model.function_changed.connect(self.on_function_changed)
        self.model.algorithm_changed.connect(self.on_algorithm_changed)
        self.model.optimization_step.connect(self.on_optimization_step)
        self.model.optimization_finished.connect(self.on_optimization_finished)
        
        # Сигналы от View - функции
        self.view.function_widget.function_changed.connect(self.on_function_selected)
        
        # Сигналы от View - алгоритмы
        self.view.algorithm_widget.algorithm_changed.connect(self.on_algorithm_selected)
        self.view.algorithm_widget.run_optimization.connect(self.on_run_optimization)
        self.view.algorithm_widget.run_optimization_random.connect(self.on_run_optimization_random)
        self.view.algorithm_widget.animation_speed_changed.connect(self.on_animation_speed_changed)
        
    def initialize(self):
        """Инициализирует приложение"""
        # Устанавливаем начальную функцию
        func_name = self.model.get_available_functions()[0]
        params_info = self.model.get_function_params_info(func_name)
        params = {p['name']: p['default'] for p in params_info}
        self.model.set_function(func_name, params)
        
        # Устанавливаем начальный алгоритм
        algo_name = self.model.get_available_algorithms()[0]
        params_info = self.model.get_algorithm_params_info(algo_name)
        params = {p['name']: p['default'] for p in params_info}
        self.model.set_algorithm(algo_name, params)
        
    def on_function_selected(self, func_name, params):
        """Обработчик выбора функции"""
        self.model.set_function(func_name, params)
        
    def on_algorithm_selected(self, algo_name, params):
        """Обработчик выбора алгоритма"""
        self.model.set_algorithm(algo_name, params)
        
    def on_function_changed(self):
        """Обработчик изменения функции в модели"""
        # Обновляем график
        X, Y, Z = self.model.get_function_data()
        self.view.plot_widget.update_function(X, Y, Z)
        
        # Очищаем путь оптимизации
        self.view.plot_widget.clear_path()
        
    def on_algorithm_changed(self):
        """Обработчик изменения алгоритма в модели"""
        pass
        
    def on_optimization_step(self, path):
        """Обработчик шага оптимизации"""
        # Обновляем путь на графике
        self.view.plot_widget.update_path(path)
        
    def on_optimization_finished(self):
        """Обработчик завершения оптимизации"""
        # Останавливаем таймер
        self.animation_timer.stop()
        print(f"Оптимизация завершена! Финальная точка: {self.model.optimization_path[-1]}")
        
    def on_animation_step(self):
        """Обработчик тика таймера анимации"""
        # Выполняем один шаг оптимизации
        if not self.model.step_optimization():
            # Оптимизация завершена
            self.animation_timer.stop()
        
    def on_run_optimization(self):
        """Обработчик запуска оптимизации из центра"""
        # Останавливаем предыдущую анимацию
        self.animation_timer.stop()
        
        # Очищаем путь
        self.view.plot_widget.clear_path()
        
        # Запускаем новую оптимизацию
        x0, y0 = 0.0, 0.0
        self.model.start_optimization(x0, y0)
        
        # Запускаем анимацию
        self.animation_timer.start(self.animation_speed)
        
    def on_run_optimization_random(self):
        """Обработчик запуска оптимизации из случайной точки"""
        # Останавливаем предыдущую анимацию
        self.animation_timer.stop()
        
        # Очищаем путь
        self.view.plot_widget.clear_path()
        
        # Генерируем случайную точку
        import numpy as np
        x_range = self.model.x_range
        y_range = self.model.y_range
        
        x0 = np.random.uniform(x_range[0], x_range[1])
        y0 = np.random.uniform(y_range[0], y_range[1])
        
        # Запускаем новую оптимизацию
        self.model.start_optimization(x0, y0)
        
        # Запускаем анимацию
        self.animation_timer.start(self.animation_speed)
        
    def on_animation_speed_changed(self, speed_ms):
        """Обработчик изменения скорости анимации"""
        self.animation_speed = speed_ms
        # Если анимация запущена, обновляем интервал таймера
        if self.animation_timer.isActive():
            self.animation_timer.setInterval(speed_ms)