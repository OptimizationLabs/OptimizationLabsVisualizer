import numpy as np
from .genetic_algorithm import genetic_algorithm
from .gradient_descent import gradient_descent

def hybrid_algorithm(
    func, 
    x_range: tuple, 
    y_range: tuple, 
    # GA params
    pop_size: int = 50,
    die_size: int = 25,  # survived_size
    p_mutation: float = 0.1,
    ga_generations: int = 25, # was 100
    # GD params
    t: float = 0.1,
    eps: float = 0.1,
    eps1: float = 1e-4,
    eps2: float = 1e-6,
    M: int = 25, # was 100
    # Hybrid params
    num_refine_best: int = 5,  # how many best from GA to refine
    clear_view=None,
    **func_params
):
    """
    Гибридный алгоритм: Генетический (препроцессор) -> Градиентный спуск (постпроцессор)
    Последовательная гибридизация типа препроцессор/постпроцессор
    """
    # Phase 1: Genetic Algorithm (global search)
    ga_gen = genetic_algorithm(
        func, x_range, y_range, 
        pop_size, die_size, p_mutation, ga_generations, 
        clear_view, **func_params
    )
    
    final_population = None
    for ga_step in ga_gen:
        # Yield GA steps for visualization
        yield ga_step  # list of points from GA
        final_population = ga_step  # last one is the final sorted population
    
    if final_population is None or len(final_population) == 0:
        # Fallback
        yield [ (np.random.uniform(*x_range), np.random.uniform(*y_range)) ]
        return
    
    # Sort final population by fitness (minimization)
    sorted_pop = sorted(
        final_population, 
        key=lambda p: func(*p, **func_params)
    )
    
    # Take top N best individuals for local refinement
    to_refine = sorted_pop[:max(1, min(num_refine_best, len(sorted_pop)))]
    
    # Phase 2: Gradient Descent from each best point
    all_refined = []
    for start_point in to_refine:
        x0, y0 = start_point
        try:
            gd_gen = gradient_descent(
                func, x0, y0, t, eps, eps1, eps2, M, **func_params
            )
            for gd_step in gd_gen:
                # Yield GD refinement steps
                yield [gd_step[0]]  # single point from GD
                all_refined.append(gd_step[0])
        except Exception:
            # If GD fails, keep original
            all_refined.append(start_point)
    
    # Final yield with best refined point
    if all_refined:
        best_refined = min(all_refined, key=lambda p: func(*p, **func_params))
        yield [best_refined]
    else:
        yield [sorted_pop[0]]