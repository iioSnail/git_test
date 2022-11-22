import torch

from model.MDCSpell import MDCSpellModel
from train_base import TrainBase


class Train(TrainBase):

    def __init__(self):
        super(Train, self).__init__()

    def init_model(self):
        return MDCSpellModel(self.args)

    def init_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=2e-5)


if __name__ == '__main__':
    train = Train()
    train.train()
