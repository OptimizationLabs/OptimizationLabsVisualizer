"""Controller для MVC приложения"""
from PySide6.QtCore import QObject, QTimer
import os


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
        
        # Путь к файлу результатов
        results_folder = "logs"
        os.makedirs(results_folder, exist_ok=True)
        self.results_file = os.path.join(results_folder, "results.txt")
        
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
        self.model.optimization_result.connect(self.on_algorithm_result_coords)
        
        # Сигналы от View
        self.view.function_widget.function_changed.connect(self.on_function_selected)
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
        
    def _save_result(self, algo_name: str, x: float, y: float, value: float):
        """Сохраняет результат оптимизации в файл"""
        try:
            # Формируем строку параметров
            params = self.model.algorithm_params
            param_str = ", ".join([f"{k}={v}" for k, v in sorted(params.items())])
            
            # Формируем строку результата
            line = (
                f"{algo_name} | "
                f"{param_str} | "
                f"x={x:.6f}, y={y:.6f}, f={value:.8f}\n"
            )
            
            with open(self.results_file, "a", encoding="utf-8") as f:
                f.write(line)
            
        except Exception as e:
            print(f"Ошибка при сохранении результата: {e}")
        
    def on_function_selected(self, func_name, params):
        """Обработчик выбора функции"""
        self.model.set_function(func_name, params)
        
    def on_algorithm_selected(self, algo_name, params):
        """Обработчик выбора алгоритма"""
        self.model.set_algorithm(algo_name, params)
        
    def on_function_changed(self):
        """Обработчик изменения функции в модели"""
        X, Y, Z = self.model.get_function_data()
        self.view.plot_widget.update_function(X, Y, Z)
        self.view.plot_widget.clear_path()
        
    def on_algorithm_changed(self):
        """Обработчик изменения алгоритма в модели"""
        pass
        
    def on_optimization_step(self, path):
        """Обработчик шага оптимизации"""
        self.view.plot_widget.update_path(path)
        
    def on_optimization_finished(self):
        """Обработчик завершения оптимизации"""
        self.animation_timer.stop()
    
    def on_algorithm_result_coords(self, result_coords: tuple):
        """Обработчик результата оптимизации"""
        x, y, z = result_coords
        self.view.algorithm_widget.on_algorithm_result_coords(x, y, z)
        
        # Сохраняем результат в файл
        algo_name = self.model.current_algorithm
        self._save_result(algo_name, x, y, z)
        
    def on_animation_step(self):
        """Обработчик тика таймера анимации"""
        if not self.model.step_optimization():
            self.animation_timer.stop()
        
    def on_run_optimization(self):
        """Обработчик запуска оптимизации из центра"""
        self.animation_timer.stop()
        self.view.plot_widget.clear_path()
        x0, y0 = 0.0, 0.0
        self.model.start_optimization(x0, y0)
        self.animation_timer.start(self.animation_speed)
        
    def on_run_optimization_random(self):
        """Обработчик запуска оптимизации из случайной точки"""
        self.animation_timer.stop()
        self.view.plot_widget.clear_path()
        x0, y0 = self.model.gen_random_point()
        self.model.start_optimization(x0, y0)
        self.animation_timer.start(self.animation_speed)
        
    def on_animation_speed_changed(self, speed_ms):
        """Обработчик изменения скорости анимации"""
        self.animation_speed = speed_ms
        if self.animation_timer.isActive():
            self.animation_timer.setInterval(speed_ms)