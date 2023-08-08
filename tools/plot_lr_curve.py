import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

# 绘制学习率变化曲线
from transformers import get_linear_schedule_with_warmup


def plot_lr_curve():
    lrs = []    # 记录学习率变化过程

    # 定义一个简单的模型，用于测试
    class SimpleModel(nn.Module):

        def __init__(self):
            super(SimpleModel, self).__init__()
            self.linear = nn.Linear(1, 1)

        def forward(self):
            return self.linear(torch.rand(4, 1))

    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-5)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=9000 * 5
    )

    for step in range(9000):
        loss = nn.functional.mse_loss(model.forward(), torch.rand(4, 1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        lrs.append(optimizer.param_groups[0]['lr'])

        scheduler.step()

    plt.plot(np.arange(0, len(lrs)), lrs)
    plt.xlabel("step")
    plt.ylabel("learning rate")
    plt.show()

    print("最大学习率为:", max(lrs))
    print("最小学习率为:", min(lrs))

    return lrs

# 使用样例

plot_lr_curve()
