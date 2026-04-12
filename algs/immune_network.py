import numpy as np


def immune_network(
    func,
    x_range: tuple,
    y_range: tuple,
    pop_size: int = 50,
    n_b: int = 10,
    n_c: int = 5,
    n_d: int = 5,
    alpha: float = 0.5,
    generations: int = 50,
    clear_view=None,
    **func_params
):
    """Алгоритм искусственной иммунной сети"""
    # Инициализация популяции
    population = [
        (
            np.random.uniform(x_range[0], x_range[1]),
            np.random.uniform(y_range[0], y_range[1])
        )
        for _ in range(pop_size)
    ]

    # Начальный шаг
    yield sorted(population, key=lambda p: func(*p, **func_params), reverse=True)

    for _ in range(generations):
        if clear_view:
            clear_view()

        # 1) Выбираем n_b лучших (максимальная BG-аффинность = минимальное f)
        fitnesses = [func(*p, **func_params) for p in population]
        sorted_idx = np.argsort(fitnesses)
        best_antibodies = [population[i] for i in sorted_idx[:n_b]]

        # 2) Клонирование
        clones = []
        for ab in best_antibodies:
            clones.extend([ab] * n_c)

        # 3) Мутация каждого клона
        mutated_clones = []
        for x, y in clones:
            dx = alpha * np.random.uniform(-0.5, 0.5)
            dy = alpha * np.random.uniform(-0.5, 0.5)
            nx = max(x_range[0], min(x + dx, x_range[1]))
            ny = max(y_range[0], min(y + dy, y_range[1]))
            mutated_clones.append((nx, ny))

        # 4) Отбор n_d лучших клонов
        clone_f = [func(*p, **func_params) for p in mutated_clones]
        best_clone_idx = np.argsort(clone_f)[:n_d]
        best_clones = [mutated_clones[i] for i in best_clone_idx]

        # 5) Объединение + сжатие (сохраняем лучшие pop_size решений)
        combined = population + best_clones
        combined_f = [func(*p, **func_params) for p in combined]
        best_combined_idx = np.argsort(combined_f)[:pop_size]
        population = [combined[i] for i in best_combined_idx]

        # Возвращаем популяцию (последняя точка — лучшая)
        yield sorted(population, key=lambda p: func(*p, **func_params), reverse=True)