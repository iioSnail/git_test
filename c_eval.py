import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from model.BertCorrectionModel import BertCorrectionModel
from model.MDCSpell import MDCSpellModel
from utils.dataset import CSCTestDataset
from train import Train
from utils.utils import save_obj, render_color_for_text, compare_text, restore_special_tokens


class Evaluation(object):

    def __init__(self):
        self.args = self.parse_args()
        self.test_set = CSCTestDataset(self.args)

        if self.args.model == 'MDCSpell':
            self.model = MDCSpellModel(self.args).eval()
        elif self.args.model == 'ChineseBertModel':
            from model.ChineseBertModel import ChineseBertModel
            self.model = ChineseBertModel(self.args).eval()
        elif self.args.model == 'Bert':
            self.model = BertCorrectionModel(self.args).eval()
        else:
            raise Exception("Unknown model: " + str(self.args.model))

        # self.model.load_state_dict(torch.load(self.args.model_path, map_location='cpu'))
        self.model.to(self.args.device)

        self.error_sentences = []

    def evaluate(self):
        self.character_level_metrics()
        self.sentence_level_metrics()
        # save_obj(self.error_sentences, self.args.output_path / self.args.test_data.name.replace(".pkl", ".result.pkl"))
        # self.print_error_sentences()

    def sentence_level_metrics(self):
        sent_P, correct_sent_TP, sent_N, detect_sent_TP = 0, 0, 0, 0
        d_precision, d_recall, d_f1 = 0, 0, 0
        c_precision, c_recall, c_f1 = 0, 0, 0

        progress = tqdm(range(len(self.test_set)), desc='Evaluation')
        for i in progress:
            src, tgt = self.test_set.__getitem__(i)
            c_output = self.model.predict(src)
            c_output = restore_special_tokens(src, c_output)

            detection_targets = torch.tensor(compare_text(src, tgt)).int().to(self.args.device)
            detection_outputs = torch.tensor(compare_text(src, c_output)).int().to(self.args.device)
            correction_outputs = torch.tensor(compare_text(c_output, tgt)).int().to(self.args.device)

            if detection_outputs.sum() > 0:
                sent_P += 1
                if correction_outputs.sum() == 0:
                    correct_sent_TP += 1

            if detection_targets.sum() > 0:
                sent_N += 1

                if (detection_outputs != detection_targets).sum() == 0:
                    detect_sent_TP += 1


            d_precision = detect_sent_TP * 1.0 / (sent_P + 1e-8)
            d_recall = detect_sent_TP * 1.0 / (sent_N + 1e-8)
            d_f1 = 2 * d_precision * d_recall / (d_precision + d_recall + 1e-8)

            c_precision = correct_sent_TP * 1.0 / (sent_P + 1e-8)
            c_recall = correct_sent_TP * 1.0 / (sent_N + 1e-8)
            c_f1 = 2 * c_precision * c_recall / (c_precision + c_recall + 1e-8)

            progress.set_postfix({
                'd_precision': d_precision,
                'd_recall': d_recall,
                'd_f1_score': d_f1,
                'c_precision': c_precision,
                'c_recall': c_recall,
                'c_f1_score': c_f1,
            })

        print("Detection Sentence-level Precision {}, Recall {}, F1_Score {}".format(d_precision, d_recall, d_f1))
        print("Correction Sentence-level Precision {}, Recall {}, F1_Score {}".format(c_precision, c_recall, c_f1))

    def character_level_metrics(self):
        d_tp, d_fp, d_tn, d_fn = 0, 0, 0, 0
        c_tp, c_fp, c_tn, c_fn = 0, 0, 0, 0

        progress = tqdm(range(len(self.test_set)), desc='Evaluation')
        for i in progress:
            src, tgt = self.test_set.__getitem__(i)
            src, tgt = src.replace(" ", ""), tgt.replace(" ", "")
            c_output = self.model.predict(src)
            c_output = restore_special_tokens(src, c_output)

            detection_targets = torch.tensor(compare_text(src, tgt)).int().to(self.args.device)
            detection_outputs = torch.tensor(compare_text(src, c_output)).int().to(self.args.device)
            correction_outputs = torch.tensor(compare_text(c_output, tgt)).int().to(self.args.device)

            d_tp += (detection_outputs[detection_targets == 1] == 1).sum().item()
            d_fp_ = (detection_outputs[detection_targets == 0] == 1).sum().item()
            d_fp += d_fp_
            d_tn += (detection_outputs[detection_targets == 0] == 0).sum().item()
            d_fn_ = (detection_outputs[detection_targets == 1] == 0).sum().item()
            d_fn += d_fn_

            c_tp += (correction_outputs[detection_targets == 1] == 0).sum().item()
            c_fp_ = (detection_outputs[detection_targets == 0] == 1).sum().item()
            c_fp += c_fp_
            c_tn += (detection_outputs[detection_targets == 0] == 0).sum().item()
            c_fn_ = (detection_outputs[detection_targets == 1] == 0).sum().item()
            c_fn += c_fn_

            d_precision, d_recall, d_f1 = Train.compute_matrix(d_tp, d_fp, d_tn, d_fn)
            c_precision, c_recall, c_f1 = Train.compute_matrix(c_tp, c_fp, c_tn, c_fn)

            progress.set_postfix({
                'd_precision': d_precision,
                'd_recall': d_recall,
                'd_f1_score': d_f1,
                'c_precision': c_precision,
                'c_recall': c_recall,
                'c_f1_score': c_f1,
            })

            if d_fp_ or d_fn_ or c_fp_ or c_fn_:
                self.error_sentences.append({
                    "source": render_color_for_text(src, compare_text(src, tgt), 'yellow'),
                    "target": render_color_for_text(tgt, compare_text(src, tgt), 'green'),
                    "correct": render_color_for_text(c_output, compare_text(src, c_output), 'red')
                })

        print("Detection Character-level Precision {}, Recall {}, F1_Score {}".format(
            *Train.compute_matrix(d_tp, d_fp, d_tn, d_fn)))
        print("Correction Character-level Precision {}, Recall {}, F1_Score {}".format(
            *Train.compute_matrix(c_tp, c_fp, c_tn, c_fn)))

    def spell_gcn_matrix(self):
        with_error = False

        # Compute F1-score
        detect_TP, detect_FP, detect_FN = 0, 0, 0
        correct_TP, correct_FP, correct_FN = 0, 0, 0
        detect_sent_TP, sent_P, sent_N, correct_sent_TP = 0, 0, 0, 0
        dc_TP, dc_FP, dc_FN = 0, 0, 0

        progress = tqdm(range(len(self.test_set)), desc='Evaluation')
        for idx in progress:
            src, tgt = self.test_set.__getitem__(idx)
            c_output = self.model.predict(src)
            c_output = restore_special_tokens(src, c_output)

            pred, actual = c_output, tgt
            pred_tokens = list(pred)
            actual_tokens = list(actual)
            detect_actual_tokens = [int(actual_token.strip(",")) \
                                    for i, actual_token in enumerate(actual_tokens) if i % 2 == 0]
            correct_actual_tokens = [actual_token.strip(",") \
                                     for i, actual_token in enumerate(actual_tokens) if i % 2 == 1]
            detect_pred_tokens = [int(pred_token.strip(",")) \
                                  for i, pred_token in enumerate(pred_tokens) if i % 2 == 0]
            _correct_pred_tokens = [pred_token.strip(",") \
                                    for i, pred_token in enumerate(pred_tokens) if i % 2 == 1]

            max_detect_pred_tokens = detect_pred_tokens

            correct_pred_zip = zip(detect_pred_tokens, _correct_pred_tokens)
            correct_actual_zip = zip(detect_actual_tokens, correct_actual_tokens)

            if detect_pred_tokens[0] != 0:
                sent_P += 1
                if sorted(correct_pred_zip) == sorted(correct_actual_zip):
                    correct_sent_TP += 1
            if detect_actual_tokens[0] != 0:
                if sorted(detect_actual_tokens) == sorted(detect_pred_tokens):
                    detect_sent_TP += 1
                sent_N += 1

            if detect_actual_tokens[0] != 0:
                detect_TP += len(set(max_detect_pred_tokens) & set(detect_actual_tokens))
                detect_FN += len(set(detect_actual_tokens) - set(max_detect_pred_tokens))
            detect_FP += len(set(max_detect_pred_tokens) - set(detect_actual_tokens))

            correct_pred_tokens = []
            # Only check the correct postion's tokens
            for dpt, cpt in zip(detect_pred_tokens, _correct_pred_tokens):
                if dpt in detect_actual_tokens:
                    correct_pred_tokens.append((dpt, cpt))

            correction_list = [actual.split(" ")[0].strip(",")]
            for dat, cpt in correct_pred_tokens:
                correction_list.append(str(dat))
                correction_list.append(cpt)

            correct_TP += len(set(correct_pred_tokens) & set(zip(detect_actual_tokens, correct_actual_tokens)))
            correct_FP += len(set(correct_pred_tokens) - set(zip(detect_actual_tokens, correct_actual_tokens)))
            correct_FN += len(set(zip(detect_actual_tokens, correct_actual_tokens)) - set(correct_pred_tokens))

            # Caluate the correction level which depend on predictive detection of BERT
            dc_pred_tokens = zip(detect_pred_tokens, _correct_pred_tokens)
            dc_actual_tokens = zip(detect_actual_tokens, correct_actual_tokens)
            dc_TP += len(set(dc_pred_tokens) & set(dc_actual_tokens))
            dc_FP += len(set(dc_pred_tokens) - set(dc_actual_tokens))
            dc_FN += len(set(dc_actual_tokens) - set(dc_pred_tokens))

        detect_precision = detect_TP * 1.0 / (detect_TP + detect_FP)
        detect_recall = detect_TP * 1.0 / (detect_TP + detect_FN)
        detect_F1 = 2. * detect_precision * detect_recall / ((detect_precision + detect_recall) + 1e-8)

        correct_precision = correct_TP * 1.0 / (correct_TP + correct_FP)
        correct_recall = correct_TP * 1.0 / (correct_TP + correct_FN)
        correct_F1 = 2. * correct_precision * correct_recall / ((correct_precision + correct_recall) + 1e-8)

        dc_precision = dc_TP * 1.0 / (dc_TP + dc_FP + 1e-8)
        dc_recall = dc_TP * 1.0 / (dc_TP + dc_FN + 1e-8)
        dc_F1 = 2. * dc_precision * dc_recall / (dc_precision + dc_recall + 1e-8)

        if with_error:
            # Token-level metrics
            print("detect_precision=%f, detect_recall=%f, detect_Fscore=%f" % (
                detect_precision, detect_recall, detect_F1))
            print("correct_precision=%f, correct_recall=%f, correct_Fscore=%f" % (
                correct_precision, correct_recall, correct_F1))
            print("dc_joint_precision=%f, dc_joint_recall=%f, dc_joint_Fscore=%f" % (dc_precision, dc_recall, dc_F1))

        detect_sent_precision = detect_sent_TP * 1.0 / (sent_P)
        detect_sent_recall = detect_sent_TP * 1.0 / (sent_N)
        detect_sent_F1 = 2. * detect_sent_precision * detect_sent_recall / (
                (detect_sent_precision + detect_sent_recall) + 1e-8)

        correct_sent_precision = correct_sent_TP * 1.0 / (sent_P)
        correct_sent_recall = correct_sent_TP * 1.0 / (sent_N)
        correct_sent_F1 = 2. * correct_sent_precision * correct_sent_recall / (
                (correct_sent_precision + correct_sent_recall) + 1e-8)

        if not with_error:
            # Sentence-level metrics
            print("detect_sent_precision=%f, detect_sent_recall=%f, detect_Fscore=%f" % (
                detect_sent_precision, detect_sent_recall, detect_sent_F1))
            print("correct_sent_precision=%f, correct_sent_recall=%f, correct_Fscore=%f" % (
                correct_sent_precision, correct_sent_recall, correct_sent_F1))

    def print_error_sentences(self):
        for item in self.error_sentences:
            print("Source:", item['source'])
            print("Correct:", item['correct'])
            print("Target:", item['target'])
            print("-" * 30)

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--test-data', type=str, default="./datasets/sighan_2015_test.csv",
                            help='The file path of test data.')
        parser.add_argument('--device', type=str, default='auto',
                            help='The device for test. auto, cpu or cuda')
        parser.add_argument('--model', type=str, default='Bert',
                            help='The model name you want to evaluate.')
        parser.add_argument('--model-path', type=str, default='./c_output/csc-best-model.pt',
                            help='The model file path. e.g. "./output/csc-best-model.pt"')
        parser.add_argument('--output-path', type=str, default='./output',
                            help='The model file path. e.g. "./output')

        args = parser.parse_known_args()[0]
        print(args)

        if args.device == 'auto':
            args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            args.device = torch.device(args.device)

        args.output_path = Path(args.output_path)
        args.test_data = Path(args.test_data)

        print("Device:", args.device)

        return args


if __name__ == '__main__':
    evaluation = Evaluation()
    evaluation.evaluate()
    # evaluation.error_sentences = load_obj('output/sighan15_test_set_simplified.result.pkl')
    evaluation.print_error_sentences()
