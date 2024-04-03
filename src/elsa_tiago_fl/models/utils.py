import numpy as np

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


class OrnsteinUhlenbeckProcess:
    def __init__(self, size, theta=0.05, mu=0, sigma=1.0, decay = 10000):
        self.size = size
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
        self.decay = decay
        self.reset()

    def reset(self):
        self.state = np.ones(self.size) * self.mu

    def sample(self, step_done):
        decay_factor = min(1.0, step_done / self.decay)  
        theta = self.theta * decay_factor
        mu = self.mu
        sigma = self.sigma * decay_factor

        dx = theta * (mu - self.state) + sigma * np.random.randn(self.size)
        self.state += dx
        return self.state
