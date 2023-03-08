import warnings

from torch.optim.lr_scheduler import _LRScheduler


class PlateauScheduler(_LRScheduler):

    def __init__(self, optimizer, check_per_step=100, reduce_times=4):
        self._last_lr = [group['lr'] for group in optimizer.param_groups]
        super(PlateauScheduler, self).__init__(optimizer)
        self.check_per_step = check_per_step
        self.reduce_times = reduce_times
        self.total_step = 0
        self.total_loss = 0.
        self.last_total_loss = 999999999.

    def add_loss(self, loss):
        self.total_loss += loss
        self.total_step += 1
        if self.total_step % self.check_per_step == 0:
            if self.total_loss > self.last_total_loss:
                self._last_lr = [lr / self.reduce_times for lr in self.get_last_lr()]
            self.last_total_loss = self.total_loss
            self.total_loss = 0.

    def get_lr(self):
        return self.get_last_lr()


class WarmupExponentialLR(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma every epoch.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, gamma, last_epoch=-1, warmup_epochs=2, warmup_factor=1.0 / 3, verbose=False,
                 **kwargs):
        self.gamma = gamma
        self.warmup_method = 'linear'
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        super().__init__(optimizer, last_epoch, verbose)

    def _get_warmup_factor_at_iter(
            self, method: str, iter: int, warmup_iters: int, warmup_factor: float
    ) -> float:
        """
        Return the learning rate warmup factor at a specific iteration.
        See https://arxiv.org/abs/1706.02677 for more details.
        Args:
            method (str): warmup method; either "constant" or "linear".
            iter (int): iteration at which to calculate the warmup factor.
            warmup_iters (int): the number of warmup iterations.
            warmup_factor (float): the base warmup factor (the meaning changes according
                to the method used).
        Returns:
            float: the effective warmup factor at the given iteration.
        """
        if iter >= warmup_iters:
            return 1.0

        if method == "constant":
            return warmup_factor
        elif method == "linear":
            alpha = iter / warmup_iters
            return warmup_factor * (1 - alpha) + alpha
        else:
            raise ValueError("Unknown warmup method: {}".format(method))

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        warmup_factor = self._get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_epochs, self.warmup_factor
        )

        if self.last_epoch <= self.warmup_epochs:
            return [base_lr * warmup_factor
                    for base_lr in self.base_lrs]
        return [group['lr'] * self.gamma
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * self.gamma ** self.last_epoch
                for base_lr in self.base_lrs]
