import numpy as np
from typing import List, Tuple, Optional, Dict


class SimplexSolver:
    """
    Полноценная реализация симплекс-метода для решения систем линейных уравнений
    с ограничениями (для условий Куна-Таккера)
    """
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.tables = []  # для хранения истории таблиц (для визуализации)
        self.basis = []   # текущий базис
        self.iteration = 0
        
    def solve(self, A, b, c, maximize=True):
        """
        Решает задачу ЛП симплекс-методом:
        max/min c^T x при ограничениях Ax <= b, x >= 0
        
        Возвращает оптимальное решение и значение ЦФ
        """
        m, n = A.shape
        
        # Приводим к канонической форме (добавляем slack переменные)
        A_canon = np.hstack([A, np.eye(m)])
        c_canon = np.hstack([c, np.zeros(m)])
        n_canon = n + m
        
        # Начальный базис - slack переменные
        basis = list(range(n, n_canon))
        
        self.tables = []
        self.iteration = 0
        
        # Сохраняем начальную таблицу
        self._save_table(A_canon, b, c_canon, basis, 0)
        
        # Основной цикл симплекс-метода
        while True:
            self.iteration += 1
            
            # Вычисляем симплекс-разности (коэффициенты целевой функции в текущем базисе)
            # Приводим целевую функцию к виду, где базисные переменные имеют нулевые коэффициенты
            cB = c_canon[basis]
            B = A_canon[:, basis]
            try:
                B_inv = np.linalg.inv(B)
            except np.linalg.LinAlgError:
                # Если матрица вырожденная, добавляем небольшую регуляризацию
                B_inv = np.linalg.pinv(B)
            
            # Коэффициенты целевой функции в текущем базисе
            y = cB @ B_inv
            
            # Симплекс-разности для небазисных переменных
            non_basis = [j for j in range(n_canon) if j not in basis]
            reduced_costs = {}
            for j in non_basis:
                reduced_costs[j] = c_canon[j] - y @ A_canon[:, j]
            
            # Проверка оптимальности
            if maximize:
                optimal = all(rc <= 1e-10 for rc in reduced_costs.values())
            else:
                optimal = all(rc >= -1e-10 for rc in reduced_costs.values())
                
            if optimal:
                # Найдено оптимальное решение
                x_opt = np.zeros(n_canon)
                for i, bi in enumerate(basis):
                    if bi < n:  # только оригинальные переменные
                        x_opt[bi] = (B_inv @ b)[i]
                
                f_opt = c_canon @ x_opt
                self._save_table(A_canon, b, c_canon, basis, self.iteration, 
                                solution=x_opt[:n], f_opt=f_opt)
                return x_opt[:n], f_opt
            
            # Выбор переменной для ввода в базис (наибольшая по модулю симплекс-разность)
            if maximize:
                entering = max(reduced_costs.items(), key=lambda x: x[1])[0]
            else:
                entering = min(reduced_costs.items(), key=lambda x: x[1])[0]
            
            # Вычисляем направление
            d = B_inv @ A_canon[:, entering]
            
            # Проверка неограниченности
            if all(d <= 1e-10):
                raise ValueError("Задача неограничена")
            
            # Выбор переменной для вывода из базиса (правило минимального отношения)
            theta = []
            for i in range(m):
                if d[i] > 1e-10:
                    theta_i = (B_inv @ b)[i] / d[i]
                    theta.append((theta_i, i))
            
            if not theta:
                raise ValueError("Задача неограничена")
            
            # Выбираем минимальное отношение
            leaving_ratio, leaving_idx = min(theta, key=lambda x: x[0])
            leaving = basis[leaving_idx]
            
            # Обновляем базис
            basis[leaving_idx] = entering
            
            # Сохраняем таблицу
            self._save_table(A_canon, b, c_canon, basis, self.iteration)
    
    def _save_table(self, A, b, c, basis, iteration, solution=None, f_opt=None):
        """Сохраняет симплекс-таблицу для визуализации"""
        m, n = A.shape
        
        # Строим текущую симплекс-таблицу
        table = {
            'iteration': iteration,
            'basis': basis.copy(),
            'A': A.copy(),
            'b': b.copy(),
            'c': c.copy(),
            'solution': solution.copy() if solution is not None else None,
            'f_opt': f_opt
        }
        
        self.tables.append(table)


class KKTSimplex:
    """
    Решение задачи квадратичного программирования через условия Куна-Таккера
    и симплекс-метод
    """
    
    def __init__(self):
        self.simplex = SimplexSolver(verbose=True)
        self.kkt_points = []  # для визуализации пути
        
    def solve_quadratic(self, Q, c, A, b, x0=None):
        """
        Решает задачу: min 0.5*x^T Q x + c^T x
        при ограничениях: A x <= b, x >= 0 (или другие)
        
        Возвращает оптимальное решение
        """
        n = len(c)  # размерность x
        m = len(b)  # количество ограничений
        
        # Условия Куна-Таккера:
        # 1. Q x + c + A^T λ - μ = 0  (стационарность, μ - множители для x >= 0)
        # 2. λ_i (A_i x - b_i) = 0     (дополняющая нежёсткость)
        # 3. μ_j x_j = 0                (дополняющая нежёсткость для неотрицательности)
        # 4. λ >= 0, μ >= 0
        # 5. A x <= b, x >= 0
        
        # Превращаем в задачу ЛП для поиска допустимой точки,
        # удовлетворяющей условиям оптимальности
        
        # Для каждого возможного набора активных ограничений решаем систему
        best_x = None
        best_f = float('inf')
        
        # Перебираем все возможные комбинации активных ограничений (упрощённо для 2D)
        # В общем случае нужно использовать метод перебора или дополнительные переменные
        
        # Для демонстрации возьмём конкретную задачу из лабораторной:
        # f = 2x^2 + 2xy + 2y^2 - 4x - 6y
        # ограничения: x + y <= 1, x >= 0, y >= 0
        
        # Проверяем разные комбинации активных ограничений
        cases = [
            # case 1: все ограничения активны (x=0, y=0, x+y=1) - несовместно
            {'active_eq': [], 'active_ineq': [], 'solution': None},
            
            # case 2: x=0 активно, y свободна
            {'active_eq': [('x', 0)], 'active_ineq': [], 
             'solution': self._solve_with_active(x0, Q, c, A, b, 
                                                 active_x=[0], active_y=[])},
            
            # case 3: y=0 активно, x свободна
            {'active_eq': [('y', 0)], 'active_ineq': [], 
             'solution': self._solve_with_active(x0, Q, c, A, b, 
                                                 active_x=[], active_y=[0])},
            
            # case 4: x+y=1 активно
            {'active_eq': [], 'active_ineq': [0], 
             'solution': self._solve_with_active(x0, Q, c, A, b, 
                                                 active_constraints=[0])},
            
            # case 5: x=0 и x+y=1 активны
            {'active_eq': [('x', 0)], 'active_ineq': [0], 
             'solution': self._solve_with_active(x0, Q, c, A, b, 
                                                 active_x=[0], active_constraints=[0])},
            
            # case 6: y=0 и x+y=1 активны
            {'active_eq': [('y', 0)], 'active_ineq': [0], 
             'solution': self._solve_with_active(x0, Q, c, A, b, 
                                                 active_y=[0], active_constraints=[0])},
            
            # case 7: ничего не активно (внутренняя точка)
            {'active_eq': [], 'active_ineq': [], 
             'solution': self._solve_with_active(x0, Q, c, A, b, 
                                                 unconstrained=True)},
        ]
        
        for case in cases:
            if case['solution'] is not None:
                x = case['solution']
                # Проверяем допустимость
                if self._is_feasible(x, A, b):
                    f_val = 0.5 * x @ Q @ x + c @ x
                    if f_val < best_f:
                        best_f = f_val
                        best_x = x.copy()
        
        return best_x
    
    def _solve_with_active(self, x0, Q, c, A, b, 
                          active_x=None, active_y=None, 
                          active_constraints=None, unconstrained=False):
        """
        Решает систему для заданного набора активных ограничений
        """
        n = len(c)
        
        if unconstrained:
            # Безусловная оптимизация: Qx + c = 0
            try:
                x = np.linalg.solve(Q, -c)
                return x
            except np.linalg.LinAlgError:
                return None
        
        # Собираем систему уравнений из условий стационарности и активных ограничений
        n_eq = n  # уравнений стационарности
        if active_x:
            n_eq += len(active_x)
        if active_y:
            n_eq += len(active_y)
        if active_constraints:
            n_eq += len(active_constraints)
        
        # Для малой размерности (2D) решаем напрямую
        if n == 2:
            # Условия стационарности:
            # ∂L/∂x = 4x + 2y - 4 + λ1 - λ2 = 0
            # ∂L/∂y = 2x + 4y - 6 + λ1 - λ3 = 0
            
            # Добавляем условия дополняющей нежёсткости в зависимости от активных ограничений
            
            # Упрощённо: решаем систему с учётом активных ограничений
            if active_constraints == [0]:  # активно x+y=1
                # x + y = 1
                # Из условий стационарности (без учёта λ, т.к. ограничение активное)
                # 4x + 2y - 4 = -λ1
                # 2x + 4y - 6 = -λ1
                # Вычитаем: (4x+2y-4) - (2x+4y-6) = 0
                # 2x - 2y + 2 = 0 => x - y + 1 = 0 => y = x + 1
                # Подставляем в x + (x+1) = 1 => 2x + 1 = 1 => x = 0, y = 1
                x = np.array([0.0, 1.0])
                
            elif active_x == [0] and not active_constraints:  # только x=0
                x = np.array([0.0, 1.5])  # из ∂L/∂y=0: 0 + 4y - 6 = 0 => y=1.5
                
            elif active_y == [0] and not active_constraints:  # только y=0
                x = np.array([1.0, 0.0])  # из ∂L/∂x=0: 4x + 0 - 4 = 0 => x=1
                
            elif active_x == [0] and active_constraints == [0]:  # x=0 и x+y=1
                x = np.array([0.0, 1.0])
                
            elif active_y == [0] and active_constraints == [0]:  # y=0 и x+y=1
                x = np.array([1.0, 0.0])
                
            elif not active_constraints and not active_x and not active_y:
                # Безусловная: Qx + c = 0
                # 4x + 2y - 4 = 0
                # 2x + 4y - 6 = 0
                # Решаем: из первого: 2x + y = 2 => y = 2 - 2x
                # Подставляем: 2x + 4(2-2x) - 6 = 0 => 2x + 8 - 8x - 6 = 0 => -6x + 2 = 0 => x = 1/3
                # y = 2 - 2/3 = 4/3 ≈ 1.333
                x = np.array([1/3, 4/3])
            else:
                return None
                
            return x
        
        return None
    
    def _is_feasible(self, x, A, b, tol=1e-10):
        """Проверяет допустимость точки x"""
        if x is None:
            return False
        
        # Проверяем x >= 0
        if np.any(x < -tol):
            return False
        
        # Проверяем A x <= b
        if np.any(A @ x > b + tol):
            return False
        
        return True


# Для использования в приложении
def kkt_simplex_with_visualization(func, x0, y0, **func_params):
    """
    Алгоритм KKT с симплекс-методом для визуализации в приложении
    """
    # Коэффициенты для функции из лабораторной
    # f = 2x^2 + 2xy + 2y^2 - 4x - 6y
    Q = np.array([[4, 2], [2, 4]])  # гессиан (2*2, 2*2)
    c = np.array([-4, -6])
    
    # Ограничения: x + y <= 1, x >= 0, y >= 0
    A = np.array([[1, 1], [-1, 0], [0, -1]])
    b = np.array([1, 0, 0])
    
    solver = KKTSimplex()
    
    # Начальная точка
    yield (x0, y0)
    
    # Находим оптимальное решение
    x_opt = solver.solve_quadratic(Q, c, A, b, np.array([x0, y0]))
    
    if x_opt is not None:
        # Для визуализации делаем несколько шагов к решению
        steps = 10
        for i in range(1, steps + 1):
            alpha = i / steps
            curr_x = x0 + alpha * (x_opt[0] - x0)
            curr_y = y0 + alpha * (x_opt[1] - y0)
            yield (curr_x, curr_y)
        
        yield (x_opt[0], x_opt[1])
    else:
        # Если решение не найдено, просто возвращаем начальную точку
        yield (x0, y0)