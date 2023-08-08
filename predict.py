import torch

from utils.utils import mock_args


class Predictor(object):

    def __init__(self):
        self.model = self.load_model()

    def parse_args(self):
        return mock_args(
            device='cpu',
            ckpt_path='my_model/multimodal-sota.ckpt',
            hyper_params={
                "dropout": 0
            }
        )

    def load_model(self):
        from models.MultiModalMyModel_SOTA_TEMP import MyModel

        args = self.parse_args()
        model = MyModel(args)
        model.load_state_dict(torch.load(args.ckpt_path, map_location='cpu')['state_dict'])
        model = model.to(args.device)

        return model

    def predict(self, sentence):
        return self.model.predict(sentence)


if __name__ == '__main__':
    predictor = Predictor()
    while True:
        sent = input("请输入要修改的句子：")
        if sent == 'exit':
            break

        pred = predictor.predict(sent)
        print("句子修改后的结果为：" + pred)


