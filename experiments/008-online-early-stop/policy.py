"""Online patience rule. observe() receives one new validation score at a time."""
import math


class EarlyStop:
    def __init__(self, patience):
        self.patience = patience
        self.epoch = self.best_epoch = self.stale = 0
        self.best = math.inf

    def observe(self, val_bpb):
        if not math.isfinite(val_bpb):
            raise ValueError("Non-finite validation score")
        self.epoch += 1
        improved = val_bpb < self.best
        if improved:
            self.best = val_bpb
            self.best_epoch = self.epoch
            self.stale = 0
        else:
            self.stale += 1
        return improved, self.stale >= self.patience
