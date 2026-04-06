from algs.simplex_solver import simplex_algorithm
from algs.gradient_descent import gradient_descent
from algs.genetic_algorithm import genetic_algorithm
from algs.particle_swarm import particle_swarm
from algs.bees_algorithm import bees_algorithm

def _run_gradient_descent(model):
    func    = model.get_current_function()
    x0, y0  = model.gen_random_point()
    p       = model.algorithm_params
    return gradient_descent(
        func, x0, y0,
        p.get('t',    0.1),
        p.get('eps',  0.1),
        p.get('eps1', 1e-4),
        p.get('eps2', 1e-6),
        p.get('M',    100),
        **model.function_params,
    )


def _run_simplex(model):
    func   = model.get_current_function()
    x0, y0 = model.gen_random_point()
    return simplex_algorithm(func, x0, y0, **model.function_params)


def _run_genetic(model):
    p = model.algorithm_params
    return genetic_algorithm(
        model.get_current_function(),
        model.x_range, model.y_range,
        p['pop_size'], p['die_size'],
        p['p_mutation'], p['generations'],
        model.clear_path,
        **model.function_params,
    )

def _run_particle_swarm(model):
    p = model.algorithm_params
    return particle_swarm(
        model.get_current_function(),
        model.x_range, model.y_range,
        p['num_particles'],
        p['generations'],
        p['current_velocity_ratio'],
        p['local_velocity_ratio'],
        p['global_velocity_ratio'],
        model.clear_path,
        **model.function_params,
    )

def _run_bees_algorithm(model):
    p = model.algorithm_params
    return bees_algorithm(
        model.get_current_function(),
        model.x_range,
        model.y_range,
        p['n_scouts'],
        p['n_elite'],
        p['n_best'],
        p['n_elite_bees'],
        p['n_best_bees'],
        p['r_elite'],
        p['r_best'],
        p['max_iter'],
        p['stagnation'],
        p['eps'],
        model.clear_path,
        **model.function_params,
    )

class OptimizationAlgorithmsFacade:

    @staticmethod
    def run(algo_name: str, optimization_model):
        """Запускает алгоритм по имени"""
        entry = OptimizationAlgorithmsFacade.ALGORITHM_REGISTRY.get(algo_name)
        if entry is None:
            raise ValueError(f'Неизвестный алгоритм: {algo_name}')
        return entry['run'](optimization_model)

    @staticmethod
    def get_algorithm_params(algo_name: str) -> list:
        """Возвращает параметры алгоритма"""
        entry = OptimizationAlgorithmsFacade.ALGORITHM_REGISTRY.get(algo_name)
        if entry is None:
            raise ValueError(f'Неизвестный алгоритм: {algo_name}')
        return entry['params']

    @staticmethod
    def get_available_algorithms() -> list[str]:
        """Возвращает список доступных алгоритмов"""
        return list(OptimizationAlgorithmsFacade.ALGORITHM_REGISTRY.keys())

    ALGORITHM_REGISTRY = {
        'gradient_descent': {
            'run': _run_gradient_descent,
            'params': [
                {'name': 't',    'label': 'Начальный шаг',        'default': 0.1,  'min': 0.001, 'max': 1.0,  'step': 0.001},
                {'name': 'eps',  'label': 'Эпсилон',              'default': 0.1,  'min': 0.01,  'max': 0.5,  'step': 0.01},
                {'name': 'eps1', 'label': 'Точность по градиенту','default': 1e-4, 'min': 1e-8,  'max': 1e-2, 'step': 1e-4},
                {'name': 'eps2', 'label': 'Точность по шагу',     'default': 1e-6, 'min': 1e-8,  'max': 1e-2, 'step': 1e-4},
                {'name': 'M',    'label': 'Макс. итераций',       'default': 100,  'min': 10,    'max': 10000,'step': 10, 'type': int},
            ],
        },
        'simplex': {
            'run': _run_simplex,
            'params': [],
        },
        'genetic_algorithm': {
            'run': _run_genetic,
            'params': [
                {'name': 'pop_size',   'label': 'Размер популяции',   'default': 50,  'min': 10,  'max': 500,  'step': 10,   'type': int},
                {'name': 'die_size',   'label': 'Смертность популяции','default': 25,  'min': 10,  'max': 500,  'step': 10,   'type': int},
                {'name': 'p_mutation', 'label': 'Вероятность мутации', 'default': 0.1, 'min': 0.0, 'max': 1.0,  'step': 0.01},
                {'name': 'generations','label': 'Макс. поколений',    'default': 100, 'min': 10,  'max': 1000, 'step': 10,   'type': int},
            ],
        },
        'particle_swarm': {
            'run': _run_particle_swarm,
            'params': [
                {'name': 'num_particles', 'label': 'Размер роя', 'default': 30, 'min': 10, 'max': 200, 'step': 5, 'type': int},
                {'name': 'generations', 'label': 'Макс. итераций', 'default': 100, 'min': 10, 'max': 1000, 'step': 10, 'type': int},
                {'name': 'current_velocity_ratio', 'label': 'Коэффициент текущей скорости (k)', 'default': 1.0, 'min': 0.01, 'max': 2.0, 'step': 0.01},
                {'name': 'local_velocity_ratio', 'label': 'Локальный коэффициент (phi_p)', 'default': 2.05, 'min': 0.1, 'max': 5.0, 'step': 0.05},
                {'name': 'global_velocity_ratio', 'label': 'Глобальный коэффициент (phi_g)', 'default': 2.05, 'min': 0.1, 'max': 5.0, 'step': 0.05},
            ],
        },
        'bees_algorithm': {
            'run': _run_bees_algorithm,
            'params': [
                {'name': 'n_scouts', 'label': 'Число разведчиков', 'default': 16, 'min': 5, 'max': 100, 'step': 1, 'type': int},
                {'name': 'n_elite', 'label': 'Элитные участки', 'default': 2, 'min': 1, 'max': 10, 'step': 1, 'type': int},
                {'name': 'n_best', 'label': 'Лучшие участки', 'default': 3, 'min': 0, 'max': 20, 'step': 1, 'type': int},

                {'name': 'n_elite_bees', 'label': 'Пчёлы в элитных участках', 'default': 7, 'min': 1, 'max': 50, 'step': 1, 'type': int},
                {'name': 'n_best_bees', 'label': 'Пчёлы в лучших участках', 'default': 4, 'min': 1, 'max': 50, 'step': 1, 'type': int},

                {'name': 'r_elite', 'label': 'Радиус элитных участков', 'default': 0.4, 'min': 0.001, 'max': 2.0, 'step': 0.01},
                {'name': 'r_best', 'label': 'Радиус лучших участков', 'default': 0.2, 'min': 0.001, 'max': 2.0, 'step': 0.01},

                {'name': 'max_iter', 'label': 'Макс. итераций', 'default': 500, 'min': 10, 'max': 2000, 'step': 10, 'type': int},
                {'name': 'stagnation', 'label': 'Порог стагнации', 'default': 20, 'min': 5, 'max': 200, 'step': 1, 'type': int},

                {'name': 'eps', 'label': 'Порог улучшения (ε)', 'default': 2.0, 'min': 1e-6, 'max': 10.0, 'step': 0.01},
            ],
        },
    }
