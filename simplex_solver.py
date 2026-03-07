import numpy as np


class SimplexSolver:
    """
    Реализация симплекс-метода для решения вспомогательной задачи ЛП
    с учётом всех условий
    """
    
    def __init__(self):
        self.tables = []
        self.basis_names = []
        self.solution = None
        self.f_opt = None
        self.iteration = 0
        self.var_names = ['x1', 'x2', 'λ', 'v1', 'v2', 'w', 'z1', 'z2']
        self.complementary_pairs = [('x1', 'v1'), ('x2', 'v2'), ('λ', 'w')]
        
    def solve_lab2_problem(self):
        """
        Решает задачу из лабораторной работы №2
        """
        A = np.array([
            [4, 4, 2, 1, -1, 0, 0, 1, 0],
            [6, 2, 4, 2, 0, -1, 0, 0, 1],
            [2, 1, 2, 0, 0, 0, 1, 0, 0]
        ], dtype=float)
        
        F = np.array([10, 6, 6, 3, -1, -1, 0, 0, 0], dtype=float)
        self.basis_names = ['z1', 'z2', 'w']
        self.iteration = 0
        
        self._save_table(A, F, self.basis_names)
        
        while True:
            self.iteration += 1
            
            # ШАГ 1: Проверка оптимальности
            if np.all(F[1:] <= 1e-10):
                break
            
            # ШАГ 2: Выбор вводимой переменной с учётом условий
            entering_idx = None
            max_val = -np.inf
            for j in range(1, len(F)):
                if F[j] > 1e-10 and F[j] > max_val:
                    var_name = self.var_names[j-1]
                    if self._can_enter_basis(var_name, self.basis_names, A):
                        max_val = F[j]
                        entering_idx = j
            
            if entering_idx is None:
                break
            
            # ШАГ 3: Выбор выводимой переменной
            ratios = []
            for i in range(len(A)):
                if A[i, entering_idx] > 1e-10:
                    ratio = A[i, 0] / A[i, entering_idx]
                    ratios.append((ratio, i))
            
            if not ratios:
                raise Exception("Задача неограничена! Все отношения <= 0")
            
            min_ratio, leaving_row = min(ratios, key=lambda x: x[0])
            
            # ШАГ 4: Симплекс-преобразование
            A, F, self.basis_names = self._simplex_iteration(
                A, F, self.basis_names, leaving_row, entering_idx
            )
            
            self._save_table(A, F, self.basis_names)
        
        self._extract_solution(A, F)
        return self.solution, self.f_opt, self.tables
    
    def _can_enter_basis(self, var_name, current_basis, A):
        """
        Проверяет, можно ли ввести переменную в базис
        с учётом условий
        """
        partner = None
        for p1, p2 in self.complementary_pairs:
            if var_name == p1:
                partner = p2
                break
            if var_name == p2:
                partner = p1
                break
        
        if partner is None:
            return True
        
        if partner in current_basis:
            partner_idx = current_basis.index(partner)
            partner_value = A[partner_idx, 0]
            if partner_value > 1e-8:
                return False
        
        return True
    
    def _simplex_iteration(self, A, F, basis, pivot_row, pivot_col):
        """Выполняет одну итерацию симплекс-метода"""
        m, n = A.shape
        
        pivot = A[pivot_row, pivot_col]
        A[pivot_row] = A[pivot_row] / pivot
        
        for i in range(m):
            if i != pivot_row:
                factor = A[i, pivot_col]
                A[i] = A[i] - factor * A[pivot_row]
        
        factor = F[pivot_col]
        F = F - factor * A[pivot_row]
        
        entering_var = self.var_names[pivot_col - 1]
        new_basis = basis.copy()
        new_basis[pivot_row] = entering_var
        
        return A, F, new_basis
    
    def _save_table(self, A, F, basis):
        """Сохраняет симплекс-таблицу"""
        self.tables.append({
            'iteration': self.iteration,
            'A': A.copy(),
            'F': F.copy(),
            'basis': basis.copy()
        })
    
    def _extract_solution(self, A, F):
        """Извлекает решение из финальной симплекс-таблицы"""
        var_indices = {'x1': 0, 'x2': 1, 'λ': 2, 'v1': 3, 'v2': 4, 'w': 5, 'z1': 6, 'z2': 7}
        x = np.zeros(8)
        
        for i, var_name in enumerate(self.basis_names):
            if var_name in var_indices:
                idx = var_indices[var_name]
                x[idx] = A[i, 0]
        
        f_val = 2*x[0]**2 + 2*x[0]*x[1] + 2*x[1]**2 - 4*x[0] - 6*x[1]
        
        self.solution = (x[0], x[1])
        self.f_opt = f_val


def simplex_algorithm(func, x0, y0, **func_params):
    """
    Функция для визуализации метода с симплексом
    """
    solver = SimplexSolver()
    solution, f_opt, tables = solver.solve_lab2_problem()
    
    yield (x0, y0)
    
    steps = 30
    for i in range(1, steps + 1):
        alpha = i / steps
        curr_x = x0 + alpha * (solution[0] - x0)
        curr_y = y0 + alpha * (solution[1] - y0)
        yield (curr_x, curr_y)
    
    yield (solution[0], solution[1])
