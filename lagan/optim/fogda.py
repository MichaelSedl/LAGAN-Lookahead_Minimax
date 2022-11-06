from torch.optim import Optimizer
import torch

class fOGDA(Optimizer):
    def __init__(self, optimizer, alpha=10, increment_iterator_every=100):
        print(
            f"Using fOGDA (alpha={alpha}; increment iterator every {increment_iterator_every} step(s))."
        )
        self.optimizer = optimizer
        # otherwise 'Optimizer' functionality doesn't work
        self.defaults = self.optimizer.defaults
        self.param_groups = self.optimizer.param_groups
        self.state = self.optimizer.state
        # fOGDA parameters
        self.alpha = alpha
        self.increment_iterator_every = increment_iterator_every
        self.iteration = 0
        self.fogda_it = 2
        self.params_copy = []
        self.old_params_copy = []
        self.updates = []
        self.old_updates = []
        self.old_difference_of_updates = []

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        no_old_params = len(self.old_params_copy) == 0
        no_old_updates = len(self.old_updates) == 0
        no_old_difference_of_updates = len(self.old_difference_of_updates) == 0

        # initialise (old) parameters
        if len(self.params_copy) > 0:
            raise RuntimeError("Something bad happend here...")
        for group in self.param_groups:
            for p in group["params"]:
                self.params_copy.append(p.data.clone())
                if no_old_params:
                    self.old_params_copy.append(p.data.clone())

        # reverse engineer update from optimizer step
        self.optimizer.step()
        i = -1
        if len(self.updates) > 0:
            raise RuntimeError("Something bad happend here...")
        for group in self.param_groups:
            for p in group["params"]:
                i += 1
                # TODO might need to exclude None values
                self.updates.append(self.params_copy[i] - p.data)

        # initialise old updates and difference of updates
        if (not no_old_updates and no_old_difference_of_updates) or (
            not no_old_difference_of_updates and no_old_updates
        ):
            raise RuntimeError("Something bad happend here...")
        if no_old_updates and no_old_difference_of_updates:
            for p in self.updates:
                self.old_updates.append(p.clone())
                self.old_difference_of_updates.append(torch.zeros_like(p))

        # compute fOGDA coefficients
        theta_p = self.alpha / (self.alpha + self.fogda_it + 1)
        theta = self.alpha / (self.alpha + self.fogda_it)
        theta_m = self.alpha / (self.alpha + self.fogda_it - 1)

        # compute new weights with fOGDA update
        i = -1
        for group in self.param_groups:
            for p in group["params"]:
                i += 1
                (
                    self.old_params_copy[i],
                    self.old_updates[i],
                    self.old_difference_of_updates[i],
                    p.data,
                ) = (
                    self.params_copy[i],
                    self.updates[i],
                    self.updates[i] - self.old_updates[i],
                    self.params_copy[i]
                    + (1 - theta_p) * (self.params_copy[i] - self.old_params_copy[i])
                    - theta_p * self.updates[i]
                    - (2 - theta)
                    * (2 - theta_p)
                    * (self.updates[i] - self.old_updates[i])
                    + (2 - theta_m) * (1 - theta_p) * self.old_difference_of_updates[i],
                )

        self.iteration += 1
        if self.iteration % self.increment_iterator_every == 0:
            # update fogda iterator
            self.fogda_it += 1

        # free parameters
        self.params_copy = []
        self.updates = []

        return loss
