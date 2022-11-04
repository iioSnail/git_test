import argparse

import torch

from model.CSCModelV1 import CSCModel


class Inference(object):

    def __init__(self):
        self.args = self.parse_args()

        self.model = CSCModel().eval()
        self.model.load_state_dict(torch.load(self.args.model_path, map_location='cpu'))
        self.model.to(self.args.device)

    def inference(self):
        text = self.args.text
        outputs, detection_outputs = self.model.predict(text)

        print("原文内容:", text)
        print("检测结果:", detection_outputs)
        print("修改结果:", outputs)

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--text', type=str, default="我爱池苹果。",
                            help='The sentence that you want to correct.')
        parser.add_argument('--device', type=str, default='auto',
                            help='The device for test. auto, cpu or cuda')
        parser.add_argument('--model-path', type=str, default='./output/csc-best-model.pt',
                            help='The model file path. e.g. "./output/csc-best-model.pt"')

        args = parser.parse_known_args()[0]
        print(args)

        if args.device == 'auto':
            args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            args.device = torch.device(args.device)

        return args


if __name__ == '__main__':
    infer = Inference()
    infer.inference()
