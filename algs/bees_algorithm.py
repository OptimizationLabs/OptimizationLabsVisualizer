import numpy as np
from typing import Generator, List

class BeesAlgorithm:
    """Пчёлковый алгос"""
    def __init__(
            self,
            func,
            func_params,
            bounds:        list[tuple],
            n_scouts:      int   = 16,
            n_elite:       int   = 2,
            n_best:        int   = 3,
            n_elite_bees:  int   = 7,
            n_best_bees:   int   = 4,
            r_elite:       float = 0.2,
            r_best:        float = 0.4,
            max_iter:      int   = 500,
            stagnation:    int   = 20,
            eps:           float = 1.0,
    ):
        self.func         = func
        self.func_params  = func_params
        self.bounds       = np.array(bounds, dtype=float)
        self.dim          = len(bounds)
        self.n_scouts     = n_scouts
        self.n_elite      = n_elite
        self.n_best       = n_best
        self.n_elite_bees = n_elite_bees
        self.n_best_bees  = n_best_bees
        self.r_elite      = r_elite
        self.r_best       = r_best
        self.max_iter     = max_iter
        self.stagnation   = stagnation
        self.spot_eps     = eps
        self.stagnation_param = 3

    def _random_point(self) -> np.ndarray:
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return lo + np.random.rand(self.dim) * (hi - lo)

    def _fitness(self, p: np.ndarray):
        return self.func(*p, **self.func_params)

    def _randon_worker_point_in_spot_area(self, spot_p, r) -> np.ndarray:
        delta = np.random.uniform(-r, r, size=self.dim)
        p = np.clip(spot_p + delta, self.bounds[:, 0], self.bounds[:, 1])
        return float(p[0]), float(p[1])

    @staticmethod
    def _euclidean_distance(p1, p2):
        return np.linalg.norm(p1 - p2)

    @staticmethod
    def _midpoint(p1, p2):
        return (p1 + p2) / 2

    def _spawn_scouts(self, n_scouts: int) -> list[np.ndarray]:
        lo, hi   = self.bounds[:, 0], self.bounds[:, 1]
        selected = []

        while len(selected) < n_scouts:
            # генерируем пачку кандидатов сразу
            batch = lo + np.random.rand(max(n_scouts * 4, 64), self.dim) * (hi - lo)

            for candidate in batch:
                if not selected:
                    selected.append(candidate.astype(np.float64))
                else:
                    # расстояния до всех принятых — одна матричная операция
                    dists = np.linalg.norm(np.array(selected) - candidate, axis=1)
                    if dists.min() >= self.spot_eps:
                        selected.append(candidate.astype(np.float64))

                if len(selected) == n_scouts:
                    return selected

        return selected

    def run(self) -> Generator[List[np.float64], None, None]:
        # 1) scouts
        stagnation_cnt = 0
        iter = 0                   
        best_history: list[float] = []

        new_population = [(self._fitness(p), p) for p in self._spawn_scouts(self.n_scouts)]
        new_population.sort(key=lambda t: t[0])
        population = new_population
        yield [x[1] for x in population]

        progress = 0
        while stagnation_cnt < self.stagnation and iter < 500:
            curr_scouts  = max(1, int(self.n_scouts * (1-progress)))
            curr_elite   = max(1, int(self.n_elite * (1 - progress)))
            curr_best    = max(0, int(self.n_best  * (1 - progress)))
            curr_r_elite = max(self.r_elite * (1 - progress), 1e-6)
            curr_r_best  = max(self.r_best  * (1 - progress), 1e-6)

            iter += 1
            progress = iter / 500
            
            # 2) zones
            elite_centers = [x for _, x in population[:curr_elite]]
            best_centers  = [x for _, x in population[curr_elite:curr_elite + curr_best]]
            new_population = population[:curr_elite+curr_best] + \
                [(self._fitness(p), p) for p in self._spawn_scouts(curr_scouts)]

            # 3) send workers
            for center in elite_centers:
                for _ in range(self.n_elite_bees):
                    p = self._randon_worker_point_in_spot_area(center, curr_r_elite)
                    new_population.append((self._fitness(p), p))

            for center in best_centers:
                for _ in range(self.n_best_bees):
                    p = self._randon_worker_point_in_spot_area(center, curr_r_best)
                    new_population.append((self._fitness(p), p))

            population = new_population
            population.sort(key=lambda t: t[0])
            best_history.append(population[0][0])
            dots = [x[1] for x in population][::-1]
            yield dots

            # 4) stagnation
            if len(best_history) > 10:
                if abs(best_history[-1] - best_history[-self.stagnation_param]) < 1e-6:
                    stagnation_cnt += 1
                else:
                    stagnation_cnt = 0
                    

def bees_algorithm(
        func,
        x_bounds:      tuple,
        y_bounds:      tuple,
        n_scouts:      int   = 16,
        n_elite:       int   = 2,
        n_best:        int   = 3,
        n_elite_bees:  int   = 7,
        n_best_bees:   int   = 4,
        r_elite:       float = 0.4,
        r_best:        float = 0.2,
        max_iter:      int   = 500,
        stagnation:    int   = 20,
        eps:           float = 2.0,
        clear_view=None,
        **func_params
) -> Generator:
    alg = BeesAlgorithm(
        func=func, func_params=func_params, bounds=[x_bounds, y_bounds],
        n_scouts=n_scouts, n_elite=n_elite, n_best=n_best,
        n_elite_bees=n_elite_bees, n_best_bees=n_best_bees,
        r_elite=r_elite, r_best=r_best,
        max_iter=max_iter, stagnation=stagnation, eps=eps,
    )
    for solution in alg.run():
        if clear_view:
            clear_view()
        yield solution