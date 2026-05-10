import numpy as np

class GeneticAlgorithm:
    """Класс с алгоритмом генетической оптимизации"""
    def __init__(self, func, x_range, y_range, population_size, survived_rate, mutation_param, generations, **func_params):
        self._func = func
        self._func_params = func_params
        self._life_space = (x_range, y_range)
        self._population_size = population_size
        self._survived_rate = survived_rate
        self._mutation_param = mutation_param
        self._generations = generations
        self._population = self._initialize_population()

    def get_population_size(self):
        """Возвращает размер популяции"""
        return self._population_size

    def get_population(self):
        """Возвращает текущую популяцию"""
        return self._population

    def set_population(self, population):
        """Устанавливает новую популяцию"""
        self._population = population

    def parent_selection(self):
        """Выбор родителей для скрещивания методом рулетки с функцией приспособленности по нормализованным значениям функции"""
        parent_values = [self._func(*parent, **self._func_params) for parent in self._population]
        F_min = min(parent_values)
        F_max = max(parent_values)
        fitness_values = [self._fitness(F_i, F_min, F_max) for F_i in parent_values]
        total_fitness = sum(fitness_values)
        if total_fitness == 0:
            probabilities = [1 / self._population_size] * self._population_size
        else:
            probabilities = [f / total_fitness for f in fitness_values]
        indices = np.random.choice(
            len(self._population),
            size=self._survived_rate,
            p=probabilities,
            replace=False
        )

        survived_parents = [self._population[i] for i in indices]
        return survived_parents

    def crossover(self, parent1, parent2) -> tuple:
        """Скрещивание двух родителей для получения потомка"""
        d = 0.25
        alpha = np.random.uniform(-d, 1+d)
        return (
            (
                parent1[0] + alpha * (parent2[0] - parent1[0]),
                parent1[1] + alpha * (parent2[1] - parent1[1])
            ),
            (
                parent2[0] + alpha * (parent1[0] - parent2[0]),
                parent2[1] + alpha * (parent1[1] - parent2[1])
            )
        )

    def mutation_old(self, individual, m = 20) -> tuple:
        """Мутация особи(особенность данной реализации - сдвиг относительно прямой)"""
        x, y = individual
        if np.random.rand() < self._mutation_param:
            alpha = 0.5
            x_ratio = (self._life_space[0][1] - self._life_space[0][0])
            y_ratio = (self._life_space[1][1] - self._life_space[1][0])
            mult = np.random.choice([-1, 1])
            delta = sum(
                (1 if np.random.rand() < 1 / m else 0) * 2**(-i)
                for i in range(1, m + 1)
            )
            x += mult * alpha * x_ratio * delta
            y += mult * alpha * y_ratio * delta
            x = max(self._life_space[0][0], min(x, self._life_space[0][1]))
            y = max(self._life_space[1][0], min(y, self._life_space[1][1]))
        return x, y
    

    def mutation(self, individual, m = 20) -> tuple:
        """Мутации особей"""
        x, y = individual
        if np.random.rand() < self._mutation_param:
            alpha = 0.5
            x_ratio = (self._life_space[0][1] - self._life_space[0][0])
            y_ratio = (self._life_space[1][1] - self._life_space[1][0])
            mult_x = np.random.choice([-1, 1])
            mult_y = np.random.choice([-1, 1])
            delta_x = sum(
                (1 if np.random.rand() < 1 / m else 0) * 2**(-i)
                for i in range(1, m + 1)
            )

            delta_y = sum(
                (1 if np.random.rand() < 1 / m else 0) * 2**(-i)
                for i in range(1, m + 1)
            )
            
            x += mult_x * alpha * x_ratio * delta_x
            y += mult_y * alpha * y_ratio * delta_y
            x = max(self._life_space[0][0], min(x, self._life_space[0][1]))
            y = max(self._life_space[1][0], min(y, self._life_space[1][1]))
        return x, y
    

    @staticmethod
    def _fitness(F_i, F_min, F_max):
        """Нормализованная функция приспособленности для минимизации"""
        if F_max == F_min:
            return 1.0  # все особи одинаковы
        
        return (F_max - F_i) / (F_max - F_min)

    def _initialize_population(self):
        """Инициализирует начальную популяцию"""
        population = []
        for _ in range(self._population_size):
            x = np.random.uniform(self._life_space[0][0], self._life_space[0][1])
            y = np.random.uniform(self._life_space[1][0], self._life_space[1][1])
            population.append((x, y))
        return population

def genetic_algorithm(func, x_range: tuple, y_range: tuple, pop_size: int, survived_size: int, p_mutation: float, generations: int = 25, clear_view=None, **func_params):
    """
    Метод генетической оптимизации для квадратичного программирования
    Реализует алгоритм из лабораторной работы №3
    """
    alg_model = GeneticAlgorithm(func, x_range, y_range, pop_size, survived_size, p_mutation, generations, **func_params)
    population = alg_model.get_population()
    yield sorted(population, key=lambda x: alg_model._func(*x, **alg_model._func_params))
    for _ in range(generations):
        survived = alg_model.parent_selection()
        while len(survived) < alg_model.get_population_size():
            idx1, idx2 = np.random.choice(len(survived), size=2, replace=False)
            parent1, parent2 = survived[idx1], survived[idx2]
            child1, child2 = alg_model.crossover(parent1, parent2)
            child1 = alg_model.mutation(child1)
            child2 = alg_model.mutation(child2)
            survived.append(child1)
            if len(survived) < alg_model.get_population_size():
                survived.append(child2)
        population = survived
        alg_model.set_population(population)
        if clear_view:
            clear_view()
        yield sorted(population, key=lambda x: alg_model._func(*x, **alg_model._func_params), reverse=True)
