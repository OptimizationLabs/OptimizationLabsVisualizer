"""Главный файл приложения - точка входа"""
import sys
from PySide6.QtWidgets import QApplication
from model import OptimizationModel
from view import MainView
from controller import OptimizationController


def main():
    """Главная функция приложения"""
    # Создаем приложение
    app = QApplication(sys.argv)
    
    # Создаем компоненты MVC
    model = OptimizationModel()
    view = MainView(model)
    controller = OptimizationController(model, view)
    
    # Показываем окно
    view.show()
    
    # Запускаем приложение
    sys.exit(app.exec())


if __name__ == '__main__':
    main()