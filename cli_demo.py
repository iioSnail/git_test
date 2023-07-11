import argparse

import torch

from utils.utils import render_color_for_text, compare_text, mock_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='macbert4csc',
                        help='The model name you want to use.')
    parser.add_argument('--device', type=str, default='auto',
                        help='The device for training. auto, cpu or cuda')
    args = parser.parse_args()

    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        args.device = args.device

    return args


def load_model(model_name: str, args):
    model_name = model_name.replace(" ", "").strip().lower()
    model = None
    if model_name == "multimodalbert":
        from models.MultiModalMyModel import MyModel
        model = MyModel(mock_args(device=args.device, hyper_params={}))
        ckpt_states = torch.load("./TrainedModels/multimodal-sota.ckpt", map_location='cpu')
        model.load_state_dict(ckpt_states['state_dict'])

    if model_name == "macbert4csc":
        from models.MacBert4CSC import MacBert4CSC_Model
        model = MacBert4CSC_Model(args)

    if model is not None:
        model = model.to(args.device)

    return model


def main():
    args = parse_args()
    print("欢迎使用中文拼写纠错工具！")
    print("模型:", args.model_name)
    print("模型加载中...")
    model = load_model(args.model_name, args)
    if model is None:
        print("模型加载失败，请确报本项目支持您给出的模型名称！")
        return

    print("模型加载完成！")

    while True:
        print("-" * 20)
        sentence = input("请输入您要修改的句子：")
        if sentence.startswith("exit"):
            return

        sentence = sentence.replace(" ", "").strip()
        pred = model.predict(sentence)
        diff_index = compare_text(sentence, pred)
        print("原句子为：", render_color_for_text(sentence, diff_index, color='red'))
        print("纠错后为：", render_color_for_text(pred, diff_index, color='green'))


if __name__ == '__main__':
    main()
