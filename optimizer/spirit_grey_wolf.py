import numpy as np
import copy
# 白眼狼优化算法！


class SpiritGWO:
    def __init__(self, upper_border, lower_border, judge_func, goal, spirit_wolf, num_wolf=5, epochs=10, minimize=True,
                 initial_seed=12):
        # Judge Function and Goal
        self.judge_func, self.goal = judge_func, goal
        # lower border and upper border
        self.ub, self.lb = upper_border, lower_border
        # number of parameters to optimize
        self.dim = self.ub.shape[1]
        # wolf number and iterate number
        self.num_wolf, self.epochs = num_wolf, epochs

        # initial three wolf
        self.alpha_pos, self.beta_pos, self.delta_pos = \
            np.zeros((self.dim)), np.zeros((self.dim)), np.zeros((self.dim))
        if minimize:
            self.alpha_score, self.beta_score, self.delta_score = np.inf, np.inf, np.inf
        else:
            self.alpha_score, self.beta_score, self.delta_score = -np.inf, -np.inf, -np.inf
        # a minimization problem or not ?
        self.minimize = minimize
        # get the position of search agents
        np.random.seed(initial_seed)
        self.positions = np.random.uniform(self.lb, self.ub, (num_wolf, self.dim))
        self.convergence_curve = []
        self.wolfs_fitness = []
        # initialize the spirit wolf
        self.spirit_wolf = spirit_wolf
        self.final_affect = 1.0 / epochs
        self.tau = 2 * np.log(self.final_affect)/epochs

    def run(self):
        epoch = 0
        while epoch < self.epochs:
            wolf_fitness = []
            # Update three wolves' positions
            for i in range(self.positions.shape[0]):
                wolf = self.positions[i, :]
                # Return back the search agents that go beyond the boundaries of the search space
                for j in range(wolf.shape[0]):
                    wolf[j] = self.ub[0, j] if wolf[j] > self.ub[0, j] else wolf[j]
                    wolf[j] = self.lb[0, j] if wolf[j] < self.lb[0, j] else wolf[j]
                # calculate the fitness value

                fitness = self.judge_func(wolf)
                wolf_fitness.append(fitness)
                if self.minimize:
                    # update alpha, beta, delta
                    if fitness < self.alpha_score:
                        self.alpha_score, self.alpha_pos = fitness, copy.copy(wolf)
                        # wolf alpha will be the best leader
                    elif self.alpha_score < fitness < self.beta_score:
                        self.beta_score, self.beta_pos = fitness, copy.copy(wolf)
                    elif self.beta_score < fitness < self.delta_score and fitness > self.alpha_score:
                        self.delta_score, self.delta_pos = fitness, copy.copy(wolf)
                else:
                    if fitness > self.alpha_score:
                        self.alpha_score, self.alpha_pos = fitness, copy.copy(wolf)
                        # wolf alpha will be the best leader
                    elif self.alpha_score > fitness > self.beta_score:
                        self.beta_score, self.beta_pos = fitness, copy.copy(wolf)
                    elif self.beta_score > fitness > self.delta_score and fitness < self.alpha_score:
                        self.delta_score, self.delta_pos = fitness, copy.copy(wolf)

            # a decrease linearly from 2 to 0
            a = 2.0 - epoch * (2.0/self.epochs)
            s_a = self.relu(np.exp(self.tau*epoch) - self.final_affect)

            # update the position of search agents including omegas
            for i in range(self.positions.shape[0]):
                for j in range(self.positions.shape[1]):
                    # update the position of wolf, circle around the prey
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2

                    # update the position of wolf alpha
                    D_alpha = abs(C1 * self.alpha_pos[j] - self.positions[i, j])
                    x1 = self.alpha_pos[j] - A1 * D_alpha
                    r1, r2 = np.random.rand(), np.random.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2

                    # update the position based on wolf beta
                    D_beta = abs(C2*self.beta_pos[j] - self.positions[i, j])
                    x2 = self.beta_pos[j] - A2 * D_beta

                    r1, r2 = np.random.rand(), np.random.rand()
                    A3 = 2*a*r1 - a
                    C3 = 2*r2

                    # update the position based on wolf delta
                    D_delta = abs(C3 * self.delta_pos[j] - self.positions[i, j])
                    x3 = self.delta_pos[j] - A3 * D_delta

                    self.positions[i, j] = (x1+x2+x3)/3
                    if (self.spirit_wolf is not None) and self.spirit_wolf[j] < np.inf:
                        r = np.random.rand()
                        d_spirit = (self.spirit_wolf[j] - self.positions[i, j]) * s_a + 2 * s_a * r - s_a
                    else:
                        d_spirit = 0
                    self.positions[i, j] = self.positions[i, j] + d_spirit
            self.wolfs_fitness.append(wolf_fitness.copy())
            self.convergence_curve.append(self.alpha_score)
            epoch += 1
            print("Generation ", epoch, "finish! Best value = ", self.alpha_score)
        print("Optimization Finish !")

    def alpha_wolf(self):
        return self.alpha_pos, self.alpha_score

    @staticmethod
    def relu(x):
        return max(0.0, x)


if __name__ == '__main__':
    def judge(x):
        x1, x2, x3, x4, x5, x6 = x[0], x[1], x[2], x[3], x[4], x[5]
        return x1 * x1 + (x2 - 6) ** 2 + (x3 - 5) ** 2 + (x4 - 9) ** 2 + (x5 - 4.2) ** 2 + (x6 - 7) ** 2
    epoch_num = 20
    num_search_agent = 15
    import matplotlib.pyplot as plt
    seed = 898
    gwo = SpiritGWO(np.array([[100]*6]), np.array([[-100]*6]),
                    judge, 0, spirit_wolf=None, num_wolf=num_search_agent, epochs=epoch_num, minimize=True, initial_seed=seed)
    gwo.run()
    print(gwo.alpha_wolf())
    gwo_spirit = SpiritGWO(np.array([[100]*6]), np.array([[-100]*6]),
                    judge, 0, spirit_wolf=np.array([np.inf, np.inf, np.inf, 8, 6, np.inf]),
                           num_wolf=num_search_agent, epochs=epoch_num, minimize=True, initial_seed=seed)
    gwo_spirit.run()
    print(gwo_spirit.alpha_wolf())
    print(judge(gwo_spirit.alpha_wolf()[0]))
    x = np.arange(0, epoch_num)
    fig, ax = plt.subplots()
    ax.plot(x, gwo.convergence_curve, label='gwo')
    ax.plot(x, gwo_spirit.convergence_curve, label='gwo spirit')
    plt.legend()
    plt.show()





