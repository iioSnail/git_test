"""
绘制Probe任务的结果
"""


import matplotlib.pyplot as plt

# ReaLise模型，拼音Probe任务，有Fusion Module，模型参与训练
realise_pinyin_wo_fm_w_t = [0.5901, 0.6591, 0.6980, 0.7506, 0.7721, 0.7939, 0.8213, 0.8366, 0.8477, 0.8511, 0.8718,
                            0.8747, 0.8768, 0.8794, 0.8865, 0.8892, 0.8954, 0.8910, 0.9009, 0.9037, ]

# ReaLise模型，拼音Probe任务，有Fusion Module，模型不参与训练
realise_pinyin_wo_fm_wo_t = [0.5490, 0.5828, 0.5850, 0.6163, 0.6268, 0.6473, 0.6304, 0.6470, 0.6432, 0.6556, 0.6668,
                             0.6718, 0.6511, 0.6681, 0.6653, 0.6707, 0.6836, 0.6698, 0.6849, 0.6776, ]

realise_pinyin_w_fm_w_t = [0.5095, 0.4897, 0.4898, 0.4906, 0.4889, 0.4889, 0.4898, 0.4898, 0.4897, 0.4905, 0.4898,
                           0.4901, 0.4897, 0.4901, 0.4897, 0.4896, 0.4897, 0.4901, 0.4902, 0.4900, ]

realise_pinyin_w_fm_wo_t = [0.5956, 0.5925, 0.6098, 0.6211, 0.6336, 0.6314, 0.6544, 0.6644, 0.6546, 0.6710, 0.6648,
                            0.6738, 0.6750, 0.6690, 0.6751, 0.6769, 0.6747, 0.6802, 0.6829, 0.6838, ]

realise_zixing_w_fm_wo_t = [0.7249, 0.7399, 0.7407, 0.7496, 0.7592, 0.7515, 0.7503, 0.7673, 0.7623, 0.7581, 0.7569,
                            0.7665, 0.7631, 0.7754, 0.7654, 0.7449, 0.7608, 0.7627, 0.7542, 0.7581, ]

realise_zixing_w_fm_w_t = [0.5065, 0.5459, 0.5073, 0.5462, 0.5462, 0.5432, 0.5482, 0.5412, 0.5416, 0.5435, 0.5412,
                           0.5007, 0.5424, 0.5027, 0.5439, 0.5408, 0.5420, 0.5073, 0.4938, 0.4942, ]

realise_zixing_wo_fm_wo_t = [0.7141, 0.7168, 0.7175, 0.7334, 0.7299, 0.7291, 0.7357, 0.7438, 0.7399, 0.7326, 0.7422,
                             0.7480, 0.7388, 0.7345, 0.7361, 0.7384, 0.7503, 0.7330, 0.7345, 0.7291, ]

realise_zixing_wo_fm_w_t = [0.5038, 0.7010, 0.5046, 0.4969, 0.5034, 0.4949, 0.4965, 0.4953, 0.4957, 0.4957, 0.4957,
                            0.4969, 0.4953, 0.5034, 0.4961, 0.5023, 0.4957, 0.4969, 0.4949, 0.4969, ]


def plot_probe_result(x_map):
    fig, ax = plt.subplots(1, 1)
    for model_name, data in x_map.items():
        data = data.copy()
        data = [0.5] + data
        ax.plot(range(0, len(data)), data)

    ax.legend(x_map.keys())
    ax.set_xticks(range(0, 21))
    fig.show()

if __name__ == '__main__':
    plot_probe_result({
        "ReaLise_WF": realise_pinyin_w_fm_wo_t,
        "ReaLise_WOF": realise_pinyin_wo_fm_wo_t,
    })

    plot_probe_result({
        "ReaLise_WF": realise_zixing_w_fm_wo_t,
        "ReaLise_WOF": realise_zixing_wo_fm_wo_t,
    })
