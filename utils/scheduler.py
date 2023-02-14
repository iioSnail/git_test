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
