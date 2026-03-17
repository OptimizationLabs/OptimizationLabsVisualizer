from algs.simplex_solver import simplex_algorithm
from algs.gradient_descent import gradient_descent
from algs.genetic_algorithm import genetic_algorithm


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


class OptimizationAlgorithmsFacade:

    @staticmethod
    def run(algo_name: str, optimization_model: 'OptimizationModel'):
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
    }
