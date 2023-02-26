import copy

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel


class MDCSpellLoss(nn.Module):

    def __init__(self, coefficient=0.85):
        super(MDCSpellLoss, self).__init__()
        # 定义Correction Network的Loss函数
        self.correction_criterion = nn.CrossEntropyLoss(ignore_index=0)
        # 定义Detection Network的Loss函数，因为是二分类，所以用Binary Cross Entropy
        self.detection_criterion = nn.BCELoss()
        # 权重系数
        self.coefficient = coefficient

    def forward(self, correction_outputs, correction_targets, detection_outputs, detection_targets):
        """
        :param correction_outputs: Correction Network的输出，Shape为(batch_size, sequence_length, hidden_size)
        :param correction_targets: Correction Network的标签，Shape为(batch_size, sequence_length)
        :param detection_outputs: Detection Network的输出，Shape为(batch_size, sequence_length)
        :param detection_targets: Detection Network的标签，Shape为(batch_size, sequence_length)
        :return:
        """
        # 计算Correction Network的loss，因为Shape维度为3，所以要把batch_size和sequence_length进行合并才能计算
        correction_loss = self.correction_criterion(correction_outputs.view(-1, correction_outputs.size(2)),
                                                    correction_targets.view(-1))
        # 计算Detection Network的loss
        detection_loss = self.detection_criterion(detection_outputs, detection_targets)
        # 对两个loss进行加权平均
        return self.coefficient * correction_loss + (1 - self.coefficient) * detection_loss


class CorrectionNetwork(nn.Module):

    def __init__(self, args):
        super(CorrectionNetwork, self).__init__()

        self.args = args

        # BERT分词器，作者并没提到自己使用的是哪个中文版的bert，我这里就使用一个比较常用的
        self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        # BERT
        self.bert = AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
        # BERT的word embedding，本质就是个nn.Embedding
        self.word_embedding_table = self.bert.get_input_embeddings()
        # 预测层。hidden_size是词向量的大小，len(self.tokenizer)是词典大小
        self.dense_layer = nn.Linear(self.bert.config.hidden_size, len(self.tokenizer))

    def forward(self, inputs, word_embeddings, detect_hidden_states):
        """
        Correction Network的前向传递
        :param inputs: inputs为tokenizer对中文文本的分词结果，
                       里面包含了token对一个的index，attention_mask等
        :param word_embeddings: 使用BERT的word_embedding对token进行embedding后的结果
        :param detect_hidden_states: Detection Network输出hidden state
        :return: Correction Network对个token的预测结果。
        """
        # 1. 使用bert进行前向传递
        bert_outputs = self.bert(token_type_ids=inputs['token_type_ids'],
                                 attention_mask=inputs['attention_mask'],
                                 inputs_embeds=word_embeddings)
        # 2. 将bert的hidden_state和Detection Network的hidden state进行融合。
        hidden_states = bert_outputs['last_hidden_state'] + detect_hidden_states
        # 3. 最终使用全连接层进行token预测
        return self.dense_layer(hidden_states)


class DetectionNetwork(nn.Module):

    def __init__(self, position_embeddings, transformer_blocks, hidden_size, args):
        """
        :param position_embeddings: bert的position_embeddings，本质是一个nn.Embedding
        :param transformer: BERT的前两层transformer_block，其是一个ModuleList对象
        """
        super(DetectionNetwork, self).__init__()

        self.args = args

        self.position_embeddings = position_embeddings
        self.transformer_blocks = transformer_blocks

        # 定义最后的预测层，预测哪个token是错误的
        self.dense_layer = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, word_embeddings):
        # 获取token序列的长度，这里为128
        sequence_length = word_embeddings.size(1)
        # 生成position embedding
        position_embeddings = self.position_embeddings(torch.LongTensor(range(sequence_length)).to(self.args.device))
        # 融合work_embedding和position_embedding
        x = word_embeddings + position_embeddings
        # 将x一层一层的使用transformer encoder进行向后传递
        for transformer_layer in self.transformer_blocks:
            x = transformer_layer(x)[0]

        # 最终返回Detection Network输出的hidden states和预测结果
        hidden_states = x
        return hidden_states, self.dense_layer(hidden_states)


class MDCSpellModel(nn.Module):

    def __init__(self, args):
        super(MDCSpellModel, self).__init__()
        self.args = args
        # 构造Correction Network
        self.correction_network = CorrectionNetwork(args)
        self._init_correction_dense_layer()

        # 构造Detection Network
        # position embedding使用BERT的
        position_embeddings = self.correction_network.bert.embeddings.position_embeddings
        # 作者在论文中提到的，Detection Network的Transformer使用BERT的权重
        # 所以我这里直接克隆BERT的前两层Transformer来完成这个动作
        transformer = copy.deepcopy(self.correction_network.bert.encoder.layer[:2])
        # 提取BERT的词向量大小
        hidden_size = self.correction_network.bert.config.hidden_size

        # 构造Detection Network
        self.detection_network = DetectionNetwork(position_embeddings, transformer, hidden_size, args)

        self.tokenizer = self.correction_network.tokenizer

        self.criteria = MDCSpellLoss()

    def forward(self, inputs):
        # 先获取word embedding，Correction Network和Detection Network都要用
        word_embeddings = self.correction_network.word_embedding_table(inputs['input_ids'])
        # Detection Network进行前向传递，获取输出的Hidden State和预测结果
        hidden_states, detection_outputs = self.detection_network(word_embeddings)
        # Correction Network进行前向传递，获取其预测结果
        correction_outputs = self.correction_network(inputs, word_embeddings, hidden_states)
        # 返回Correction Network 和 Detection Network 的预测结果。
        # 在计算损失时`[PAD]`token不需要参与计算，所以这里将`[PAD]`部分全都变为0
        return correction_outputs, detection_outputs.squeeze(2) * inputs['attention_mask']

    def _init_correction_dense_layer(self):
        """
        原论文中提到，使用Word Embedding的weight来对Correction Network进行初始化
        """
        self.correction_network.dense_layer.weight.data = self.correction_network.word_embedding_table.weight.data

    def compute_loss(self, outputs, targets, inputs, *args, **kwargs):
        correction_outputs, detection_outputs = outputs
        correction_targets = targets['input_ids']
        detection_targets = (inputs['input_ids'] != targets['input_ids']).float()
        return self.criteria(correction_outputs, correction_targets, detection_outputs, detection_targets)

    def extract_outputs(self, outputs):
        return outputs[0].argmax(dim=2)

    def predict(self, sentence):
        sentence = sentence.replace(" ", "")
        sentence = ' '.join(sentence)
        tokenizer = self.correction_network.tokenizer
        inputs = tokenizer(sentence, padding=True, return_tensors='pt').to(self.args.device)
        c_outputs, _ = self.forward(inputs)
        c_output = c_outputs.argmax(2).squeeze()[1:-1]
        return tokenizer.decode(c_output).replace(" ", "")

    def d_predict(self, src, tgt=None):
        src = src.replace(" ", "")
        src = " ".join(src)
        tgt = tgt.replace(" ", "")
        tgt = " ".join(tgt)
        inputs = self.tokenizer(src, return_tensors='pt').to(self.args.device)
        targets = self.tokenizer(tgt, return_tensors='pt').to(self.args.device)

        _, d_outputs = self.forward(inputs)
        d_targets = (inputs['input_ids'] != targets['input_ids']).int().squeeze()
        d_outputs = (torch.sigmoid(d_outputs.squeeze()) > 0.5).int()
        return d_outputs[1:-1], d_targets[1:-1]
