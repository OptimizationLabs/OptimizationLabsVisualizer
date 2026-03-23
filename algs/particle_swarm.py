import numpy as np

class Particle:
    def __init__(self, swarm):
        self.current_position = self._get_init_position(swarm)
        self.local_best_position = self.current_position.copy()
        self.local_best_final_func = swarm.get_final_func(self.current_position)
        self.velocity = self._get_init_velocity(swarm)

    def _get_init_position(self, swarm):
        return np.random.rand(swarm.dimension) * (swarm.maxvalues - swarm.minvalues) + swarm.minvalues

    def _get_init_velocity(self, swarm):
        minval = -(swarm.maxvalues - swarm.minvalues)
        maxval = swarm.maxvalues - swarm.minvalues
        return np.random.rand(swarm.dimension) * (maxval - minval) + minval

    def next_iteration(self, swarm):
        rnd_current = np.random.rand(swarm.dimension)
        rnd_global = np.random.rand(swarm.dimension)

        velo_ratio = swarm.local_velocity_ratio + swarm.global_velocity_ratio
        denom = np.abs(2.0 - velo_ratio - np.sqrt(velo_ratio ** 2 - 4.0 * velo_ratio))
        common_ratio = (2.0 * swarm.current_velocity_ratio) / denom if denom > 1e-10 else 1.0

        new_v1 = common_ratio * self.velocity
        new_v2 = common_ratio * swarm.local_velocity_ratio * rnd_current * \
                 (self.local_best_position - self.current_position)
        new_v3 = common_ratio * swarm.global_velocity_ratio * rnd_global * \
                 (swarm.global_best_position - self.current_position)

        self.velocity = new_v1 + new_v2 + new_v3
        self.current_position += self.velocity

        # Зажимаем в границы области поиска
        self.current_position = np.clip(self.current_position, swarm.minvalues, swarm.maxvalues)

        final_func = swarm.get_final_func(self.current_position)
        if final_func < self.local_best_final_func:
            self.local_best_position = self.current_position.copy()
            self.local_best_final_func = final_func


class Swarm:
    def __init__(self, swarmsize, minvalues, maxvalues,
                 current_velocity_ratio, local_velocity_ratio, global_velocity_ratio,
                 func, func_params):
        self.dimension = len(minvalues)
        self.swarmsize = swarmsize
        self.minvalues = np.array(minvalues)
        self.maxvalues = np.array(maxvalues)
        self.current_velocity_ratio = current_velocity_ratio
        self.local_velocity_ratio = local_velocity_ratio
        self.global_velocity_ratio = global_velocity_ratio
        self.global_best_final_func = None
        self.global_best_position = None
        self.func = func
        self.func_params = func_params
        self.swarm_particles = [Particle(self) for _ in range(swarmsize)]

    def get_final_func(self, position):
        final_func = self.func(position[0], position[1], **self.func_params)
        if self.global_best_final_func is None or final_func < self.global_best_final_func:
            self.global_best_final_func = final_func
            self.global_best_position = position.copy()
        return final_func

    def next_iteration(self):
        for particle in self.swarm_particles:
            particle.next_iteration(self)

    def get_positions(self):
        return [(p.current_position[0], p.current_position[1]) for p in self.swarm_particles]


def particle_swarm(func, x_range, y_range, num_particles=30, generations=100,
                   current_velocity_ratio=1.0, local_velocity_ratio=2.05,
                   global_velocity_ratio=2.05, clear_view=None, **func_params):
    """Генератор для анимации"""
    minv = [x_range[0], y_range[0]]
    maxv = [x_range[1], y_range[1]]
    swarm = Swarm(num_particles, minv, maxv,
                  current_velocity_ratio, local_velocity_ratio, global_velocity_ratio,
                  func, func_params)

    # Начальное положение роя
    yield sorted(swarm.get_positions(), key=lambda p: func(*p, **func_params), reverse=True)

    for _ in range(generations):
        swarm.next_iteration()
        if clear_view:
            clear_view()
        yield sorted(swarm.get_positions(), key=lambda p: func(*p, **func_params), reverse=True)