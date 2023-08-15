import argparse

import torch

from c_train import C_Train
from utils.log_utils import log
from utils.str_utils import is_float
from utils.utils import mock_args


class Predictor(object):

    def __init__(self):
        self.args = self.parse_args()
        self.model = self.load_model()

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=str, default='bert',
                            help='The model name you want to evaluate.')
        parser.add_argument('--device', type=str, default='auto',
                            help='The device for training. auto, cpu or cuda')
        parser.add_argument('--ckpt-path', type=str, default=None, help='The filepath of checkpoint for test.')
        parser.add_argument('--hyper-params', type=str, default="",
                            help='The hyper parameters of your model. The type must be json.')
        parser.add_argument('--bert-path', type=str, help="The pretrained model path.")

        args = parser.parse_known_args()[0]

        if args.device == 'auto':
            args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            args.device = args.device

        print("Device:", args.device)

        log.info(args)

        try:
            hyper_params = {}
            for param in args.hyper_params.split(","):
                if len(param.split("=")) != 2:
                    continue

                key, value = param.split("=")
                if is_float(value):
                    value = float(value)
                    if value == int(value):
                        value = int(value)

                hyper_params[key] = value
            args.hyper_params = hyper_params
        except:
            log.error(
                "Failed to resolve hyper-params. The pattern must look like 'key=value,key=value'. hyper_params: %s" % args.hyper_params)
            exit(0)

        if len(args.hyper_params) > 0:
            print("Hyper parameters:", args.hyper_params)

        return args

    def load_model(self):
        model = C_Train.model_select(self.args)
        model.load_state_dict(torch.load(self.args.ckpt_path, map_location='cpu')['state_dict'])
        model = model.to(self.args.device)

        return model

    def predict(self, sentence):
        return self.model.predict(sentence)


if __name__ == '__main__':
    predictor = Predictor()
    while True:
        # sent = input("请输入要修改的句子：")
        sent = "辱创澜倒骨头会死人吗"
        if sent == 'exit':
            break

        pred = predictor.predict(sent)
        print("句子修改后的结果为：" + pred)
