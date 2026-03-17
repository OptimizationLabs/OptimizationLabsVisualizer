"""View для MVC приложения"""
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                                QTabWidget, QLabel, QDoubleSpinBox, QPushButton,
                                QGroupBox, QFormLayout, QSplitter, QSlider, QSpinBox)
from PySide6.QtCore import Qt, Signal
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import math
import numpy as np


class FunctionWidget(QWidget):
    """Виджет для выбора функции и её параметров"""
    
    function_changed = Signal(str, dict)
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.param_spinboxes = {}
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Вкладки для разных функций
        self.tabs = QTabWidget()
        
        # Создаем вкладки для каждой функции
        for func_name in self.model.get_available_functions():
            tab = self.create_function_tab(func_name)
            self.tabs.addTab(tab, func_name.capitalize())
            
        self.tabs.currentChanged.connect(self.on_function_changed)
        
        layout.addWidget(self.tabs)
        self.setLayout(layout)
        
    def create_function_tab(self, func_name):
        """Создает вкладку для функции"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Информация о функции
        info_label = QLabel(f"Функция: {func_name}")
        info_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(info_label)
        
        # Параметры функции
        params_info = self.model.get_function_params_info(func_name)
        
        if params_info:
            params_group = QGroupBox("Параметры")
            params_layout = QFormLayout()
            
            for param in params_info:
                spinbox = QDoubleSpinBox()
                spinbox.setMinimum(param['min'])
                spinbox.setMaximum(param['max'])
                spinbox.setSingleStep(param['step'])
                spinbox.setValue(param['default'])
                spinbox.valueChanged.connect(
                    lambda v, fn=func_name, pn=param['name']: 
                    self.on_param_changed(fn, pn, v)
                )
                
                params_layout.addRow(f"{param['name']}:", spinbox)
                
                # Сохраняем spinbox
                if func_name not in self.param_spinboxes:
                    self.param_spinboxes[func_name] = {}
                self.param_spinboxes[func_name][param['name']] = spinbox
                
            params_group.setLayout(params_layout)
            layout.addWidget(params_group)
        else:
            layout.addWidget(QLabel("Нет параметров"))
            
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def on_function_changed(self, index):
        """Обработчик изменения выбранной функции"""
        func_name = self.model.get_available_functions()[index]
        params = self.get_current_params(func_name)
        self.function_changed.emit(func_name, params)
        
    def on_param_changed(self, func_name, param_name, value):
        """Обработчик изменения параметра"""
        # Проверяем, что это текущая функция
        current_index = self.tabs.currentIndex()
        current_func = self.model.get_available_functions()[current_index]
        
        if current_func == func_name:
            params = self.get_current_params(func_name)
            self.function_changed.emit(func_name, params)
            
    def get_current_params(self, func_name):
        """Получает текущие значения параметров функции"""
        if func_name not in self.param_spinboxes:
            return {}
        
        params = {}
        for param_name, spinbox in self.param_spinboxes[func_name].items():
            params[param_name] = spinbox.value()
        return params


class AlgorithmWidget(QWidget):
    """Виджет для выбора алгоритма и его параметров"""
    
    algorithm_changed = Signal(str, dict)
    run_optimization = Signal()
    run_optimization_random = Signal()
    animation_speed_changed = Signal(int)
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.param_spinboxes = {}
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Вкладки для разных алгоритмов
        self.tabs = QTabWidget()
        
        # Создаем вкладки для каждого алгоритма
        for algo_name in self.model.get_available_algorithms():
            tab = self.create_algorithm_tab(algo_name)
            self.tabs.addTab(tab, algo_name.replace('_', ' ').title())
            
        self.tabs.currentChanged.connect(self.on_algorithm_changed)
        
        layout.addWidget(self.tabs)
        
        # Группа управления анимацией
        animation_group = QGroupBox("Управление анимацией")
        animation_layout = QVBoxLayout()
        
        # Слайдер скорости анимации
        speed_layout = QHBoxLayout()
        speed_label = QLabel("Скорость анимации:")
        speed_layout.addWidget(speed_label)
        
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(10)
        self.speed_slider.setMaximum(500)
        self.speed_slider.setValue(50) 
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        self.speed_slider.setTickInterval(50)
        self.speed_slider.valueChanged.connect(self.on_speed_changed)
        speed_layout.addWidget(self.speed_slider)
        
        self.speed_value_label = QLabel("50 мс")
        speed_layout.addWidget(self.speed_value_label)
        
        animation_layout.addLayout(speed_layout)
        animation_group.setLayout(animation_layout)
        layout.addWidget(animation_group)
        
        # Кнопки запуска оптимизации
        buttons_layout = QVBoxLayout()
        
        #self.run_button = QPushButton("Запустить из центра (0, 0)")
        #self.run_button.clicked.connect(self.run_optimization.emit)
        #buttons_layout.addWidget(self.run_button)
        
        self.run_random_button = QPushButton("Запустить оптимизацию")
        self.run_random_button.clicked.connect(self.run_optimization_random.emit)
        buttons_layout.addWidget(self.run_random_button)
        
        layout.addLayout(buttons_layout)

        # QLabel результата (в самом низу, на всю ширину)
        self.result_label = QLabel("Точка не найдена")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-weight: bold; font-size: 14px;")

        layout.addWidget(self.result_label)
        
        self.setLayout(layout)
    
    def on_algorithm_result_coords(self, x, y, value):
        """Обработчик результата алгоритма"""
        self.result_label.setText(
            f"Найдена точка: x={x:.4f}, y={y:.4f}, f(x,y)={value:.4f}"
        )
        
    def on_speed_changed(self, value):
        """Обработчик изменения скорости анимации"""
        self.speed_value_label.setText(f"{value} мс")
        self.animation_speed_changed.emit(value)
        
    def create_algorithm_tab(self, algo_name):
        """Создает вкладку для алгоритма"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Информация об алгоритме
        info_label = QLabel(f"Алгоритм: {algo_name.replace('_', ' ').title()}")
        info_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(info_label)
        
        # Параметры алгоритма
        params_info = self.model.get_algorithm_params_info(algo_name)
        
        if params_info:
            params_group = QGroupBox("Параметры")
            params_layout = QFormLayout()

            for param in params_info:

                param_type = param.get('type', float)

                if param_type is int:
                    spinbox = QSpinBox()
                    spinbox.setMinimum(param['min'])
                    spinbox.setMaximum(param['max'])
                    spinbox.setSingleStep(param['step'])
                    spinbox.setValue(param['default'])

                else:
                    spinbox = QDoubleSpinBox()
                    spinbox.setMinimum(param['min'])
                    spinbox.setMaximum(param['max'])
                    spinbox.setSingleStep(param['step'])
                    spinbox.setValue(param['default'])

                    # Автоматическое вычисление количества знаков
                    step = param['step']
                    if step < 1:
                        decimals = max(0, -int(math.floor(math.log10(step))))
                    else:
                        decimals = 0

                    spinbox.setDecimals(decimals)

                spinbox.valueChanged.connect(
                    lambda v, an=algo_name, pn=param['name']:
                    self.on_param_changed(an, pn, v)
                )

                params_layout.addRow(f"{param.get('label', param['name'])}:", spinbox)

                if algo_name not in self.param_spinboxes:
                    self.param_spinboxes[algo_name] = {}

                self.param_spinboxes[algo_name][param['name']] = spinbox
                
            params_group.setLayout(params_layout)
            layout.addWidget(params_group)
            
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def on_algorithm_changed(self, index):
        """Обработчик изменения выбранного алгоритма"""
        algo_name = self.model.get_available_algorithms()[index]
        params = self.get_current_params(algo_name)
        self.algorithm_changed.emit(algo_name, params)
        
    def on_param_changed(self, algo_name, param_name, value):
        """Обработчик изменения параметра"""
        # Проверяем, что это текущий алгоритм
        current_index = self.tabs.currentIndex()
        current_algo = self.model.get_available_algorithms()[current_index]
        
        if current_algo == algo_name:
            params = self.get_current_params(algo_name)
            self.algorithm_changed.emit(algo_name, params)
            
    def get_current_params(self, algo_name):
        """Получает текущие значения параметров алгоритма"""
        if algo_name not in self.param_spinboxes:
            return {}
        
        params = {}
        for param_name, spinbox in self.param_spinboxes[algo_name].items():
            params[param_name] = spinbox.value()
        return params


class PlotWidget(QWidget):
    """Виджет для отображения 3D графика"""

    @staticmethod
    def scale_to_range(arr: np.ndarray, z_range=None, ref_min=None, ref_max=None) -> np.ndarray:
        """Масштабирует массив arr к диапазону z_range (min, max)
        Args:
            arr: массив для масштабирования
            z_range: целевой диапазон (min, max)
            ref_min: опциональное reference минимальное значение (если None, используется min(arr))
            ref_max: опциональное reference максимальное значение (если None, используется max(arr))
        """
        z_min, z_max = z_range

        # Используем reference значения, если они предоставлены
        arr_min = ref_min if ref_min is not None else np.min(arr)
        arr_max = ref_max if ref_max is not None else np.max(arr)

        if arr_max == arr_min:
            return np.full_like(arr, z_min)

        scaled = (arr - arr_min) / (arr_max - arr_min)
        scaled = scaled * (z_max - z_min) + z_min
        return scaled

    def __init__(self):
        super().__init__()
        self.current_X = None
        self.current_Y = None
        self.current_Z = None
        self.z_min = None  # Сохраняем min для масштабирования
        self.z_max = None  # Сохраняем max для масштабирования
        self.path_items = []
        self.default_normalization_scaling = (0, 10)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Создаем 3D виджет
        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=40, elevation=30, azimuth=45)

        # Добавляем сетки для осей
        self.setup_grid()

        # Элемент для отображения поверхности
        self.surface_item = None

        layout.addWidget(self.view)
        self.setLayout(layout)
        self.setMinimumSize(400, 400)
        self.view.setBackgroundColor((250, 250, 250))

    def setup_grid(self):
        """Настройка сеток координат"""
        # Сетка XY (внизу)
        gx = gl.GLGridItem()
        gx.setSpacing(1, 1, 1)
        gx.scale(1, 1, 1)
        gx.setColor((0, 0.5, 0))
        self.view.addItem(gx)

        # Оси координат
        axis = gl.GLAxisItem()
        axis.setSize(10, 10, 10)
        self.view.addItem(axis)

    def update_function(self, X, Y, Z):
        """Обновляет 3D график функции"""
        # Сохраняем текущие данные для обработки кликов
        self.current_X = X
        self.current_Y = Y
        self.current_Z = Z

        # Сохраняем min и max для последующего масштабирования точек
        self.z_min = np.min(Z)
        self.z_max = np.max(Z)

        # Удаляем старую поверхность
        if self.surface_item is not None:
            self.view.removeItem(self.surface_item)

        # Создаем цвета для поверхности
        colors = pg.colormap.get('bmy', source="colorcet").mapToFloat((Z.T - Z.min()) / (Z.max() - Z.min()))

        # Создаем новую поверхность
        self.surface_item = gl.GLSurfacePlotItem(
            x=X[0, :],
            y=Y[:, 0],
            z=PlotWidget.scale_to_range(
                self.current_Z.T,
                z_range=self.default_normalization_scaling,
                ref_min=self.z_min,
                ref_max=self.z_max
            ),
            colors=colors,
            shader='shaded',
            smooth=True
        )

        self.view.addItem(self.surface_item)

    def update_path(self, path):
        """
        path: список кортежей (x, y, z) — абсолютные координаты точек пути.
        Z-координаты уже вычислены контроллером и не зависят от текущей поверхности.
        """
        self.clear_path()

        if not path:
            return

        x_coords = np.array([p[0] for p in path])
        y_coords = np.array([p[1] for p in path])
        z_coords = np.array([p[2] for p in path])

        if self.z_min is not None and self.z_max is not None:
            z_scaled = PlotWidget.scale_to_range(
                z_coords,
                z_range=(self.default_normalization_scaling[0],
                         self.default_normalization_scaling[1]),
                ref_min=self.z_min,
                ref_max=self.z_max
            )
        else:
            z_scaled = z_coords

        pts = np.column_stack([x_coords, y_coords, z_scaled])
        if len(pts) > 1:
            scatter_path = gl.GLScatterPlotItem(
                pos=pts[:-1],
                color=(0.0, 0.5, 0.5, 1.0),
                size=6,
                pxMode=True
            )
            self.view.addItem(scatter_path)
            self.path_items.append(scatter_path)

        scatter_final = gl.GLScatterPlotItem(
            pos=pts[-1:],
            color=(1.0, 0.0, 0.0, 1.0),
            size=12,
            pxMode=True
        )
        self.view.addItem(scatter_final)
        self.path_items.append(scatter_final)

    def clear_path(self):
        """Очищает путь оптимизации"""
        for item in self.path_items:
            try:
                self.view.removeItem(item)
            except Exception:
                pass
        self.path_items.clear()


class MainView(QMainWindow):
    """Главное окно приложения"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Optimization Labs Visualizer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Основной layout
        main_layout = QHBoxLayout()
        
        # Splitter для управления размерами
        splitter = QSplitter(Qt.Horizontal)
        
        self.plot_widget = PlotWidget()
        splitter.addWidget(self.plot_widget)
        
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        
        # Виджет функций
        self.function_widget = FunctionWidget(self.model)
        right_layout.addWidget(self.function_widget)
        
        # Виджет алгоритмов
        self.algorithm_widget = AlgorithmWidget(self.model)
        right_layout.addWidget(self.algorithm_widget)
        
        right_panel.setLayout(right_layout)
        splitter.addWidget(right_panel)
        
        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)
