"""Turns a stream of CCA scores into discrete 'clicks' (debounce + margin + lockout)."""
import numpy as np


class Selector:
    def __init__(self, margin_tau=0.05, n_consecutive=3, lockout_s=1.0):
        self.tau, self.n, self.lockout = margin_tau, n_consecutive, lockout_s
        self.reset()

    def reset(self, now=0.0, lock=False):
        self.last, self.streak = None, 0
        self.locked_until = now + self.lockout if lock else 0.0

    def update(self, scores, now):
        """Returns (box, margin) when a selection fires, else None."""
        if now < self.locked_until:
            return None
        order = np.argsort(scores)[::-1]
        best = int(order[0])
        margin = float(scores[order[0]] - scores[order[1]])
        if margin > self.tau:
            self.streak = self.streak + 1 if best == self.last else 1
            self.last = best
        else:
            self.last, self.streak = None, 0
        if self.streak >= self.n:
            self.reset(now, lock=True)
            return best, margin
        return None
