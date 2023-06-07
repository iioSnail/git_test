import sys, os

sys.path.insert(1, os.path.abspath("."))
sys.path.insert(1, os.path.abspath(".."))

os.chdir(os.path.pardir)

import torch

from utils.utils import token_embeddings_visualise, mock_args


def plot_bert_embedding():
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")

    inputs = tokenizer(["我喜欢吃苹果", "我喜欢吃平果"], return_tensors='pt')
    outputs = model.bert(**inputs).last_hidden_state[:, 1:-1, :]

    token_embeddings_visualise(outputs[0], "我 喜 欢 吃 苹 果", filename="outputs/bert_pca_1.png")
    token_embeddings_visualise(outputs[1], "我 喜 欢 吃 平 果", filename="outputs/bert_pca_2.png")


def plot_mmbert_embedding():
    from models.MultiModalMyModel_SOTA import MyModel

    model = MyModel(mock_args(device='cuda', hyper_params={}))
    model.load_state_dict(torch.load("./temp/multimodal-sota.ckpt")['state_dict'])
    model = model.to("cuda").eval()

    inputs = model.tokenizer(["我喜欢吃苹果", "我喜欢吃平果"], return_tensors='pt')
    input_pinyins = model.input_helper.convert_tokens_to_pinyin_embeddings(inputs['input_ids'].view(-1))
    images = model.input_helper.convert_tokens_to_images(inputs['input_ids'].view(-1), None)

    _, hidden_states = model(inputs.to("cuda"), input_pinyins.to("cuda"), images.to("cuda"), output_hidden_states=True)
    hidden_states = hidden_states[:, 1:-1, :].detach().cpu()

    token_embeddings_visualise(hidden_states[0, :, :768], "我 喜 欢 吃 苹 果", filename="outputs/mmbert_pca_1.png")
    token_embeddings_visualise(hidden_states[1, :, :768], "我 喜 欢 吃 平 果", filename="outputs/mmbert_pca_2.png")


if __name__ == '__main__':
    # plot_bert_embedding()
    plot_mmbert_embedding()