from pathlib import Path

from utils.log_utils import log
from utils.utils import render_color_for_text, mkdir


class CSCMetrics:
    """
    与现有论文保持一致的评价指标实现
    """

    def __init__(self):
        self.total_sent_num = 0
        self.abnormal_pairs = []
        self.error_pairs = []

        self.result_pairs = []

    def add_sentence(self, src, tgt, pred):
        self.total_sent_num += 1

        src_tokens, tgt_tokens, pred_tokens = None, None, None
        if type(src) == str and type(tgt) == str and type(pred) == str:
            src = src.replace(" ", "")
            tgt = tgt.replace(" ", "")
            pred = pred.replace(" ", "")
            src_tokens = list(src)

        if not src_tokens:
            self.abnormal_pairs.append((src, tgt, pred))

        if len(src) != len(tgt) or len(tgt) != len(pred):
            self.abnormal_pairs.append((src, tgt, pred))
            return

        if pred != tgt:
            # 预测错了
            self.error_pairs.append((src, tgt, pred))

        self.result_pairs.append((src, tgt, pred))

    def _get_sent_level_detect_metrics(self):
        correct_num = 0
        true_positive = 0
        target_positive = 0
        pred_positive = 0

        def _is_full_detect(src_, target_, pred_):
            src_tokens = list(src_)
            tgt_tokens = list(target_)
            pred_tokens = list(pred_)
            for t1, t2, t3 in zip(src_tokens, tgt_tokens, pred_tokens):
                if (t1 != t2 and t1 == t3) or (t1 == t2 and t2 != t3):
                    return False

            return True

        for src, target, pred in self.result_pairs:

            true_detect = _is_full_detect(src, target, pred)

            if true_detect:
                correct_num += 1

            if src != target and true_detect:
                true_positive += 1

            if src != target:
                target_positive += 1

            if src != pred:
                pred_positive += 1

        acc = correct_num / (len(self.result_pairs) + 1e-8)
        precision = true_positive / (pred_positive + 1e-8)
        recall = true_positive / (target_positive + 1e-8)
        f1 = (2 * precision * recall) / (precision + recall + 1e-8)
        return acc, precision, recall, f1

    def _get_sent_level_correct_metrics(self):
        correct_num = 0
        true_positive = 0
        target_positive = 0
        pred_positive = 0

        for src, target, pred in self.result_pairs:
            if target == pred:
                correct_num += 1

            if src != target and target == pred:
                # 原句子有错，且纠对了
                true_positive += 1

            if src != target:
                # 原句子有错
                target_positive += 1

            if src != pred:
                # 对原句子进行了纠错（原句子是否有错不管）
                pred_positive += 1

        acc = correct_num / (len(self.result_pairs) + 1e-8)
        precision = true_positive / (pred_positive + 1e-8)
        recall = true_positive / (target_positive + 1e-8)
        f1 = (2 * precision * recall) / (precision + recall + 1e-8)
        return acc, precision, recall, f1

    def print_results(self):
        log.info("------------------------------------------------------------")
        log.info("Sentence-level Detect Acc: %.4f, P: %.4f, R: %.4f, F1: %.4f"
                 % self._get_sent_level_detect_metrics())
        log.info("Sentence-level Correct Acc: %.4f, P: %.4f, R: %.4f, F1: %.4f"
                 % self._get_sent_level_correct_metrics())
        log.info("------------------------------------------------------------")

        log.info("Total Sentences Num: %d, Error Sentences Num: %d" % (self.total_sent_num, len(self.abnormal_pairs)))

    def print_errors(self):
        for src, tgt, pred in self.error_pairs:
            print("---------------------------")
            src_tokens = list(src)
            tgt_tokens = list(tgt)
            pred_tokens = list(pred)
            tgt_detects = [1 if src_tokens[i] != tgt_tokens[i] else 0 for i in range(len(src_tokens))]
            pred_detects = [1 if src_tokens[i] != pred_tokens[i] else 0 for i in range(len(src_tokens))]
            print("src : %s" % src)
            print("tgt : %s" % render_color_for_text(tgt, tgt_detects, 'green'))
            print("pred: %s" % render_color_for_text(pred, pred_detects, 'red'))

    def print_abnormal_pairs(self):
        for src, tgt, pred in self.abnormal_pairs:
            print("*********************************")
            print("src : %s" % src)
            print("tgt : %s" % tgt)
            print("pred: %s" % pred)


    def export_sigan_format(self, dir_path='outputs'):
        abnormal_pairs = []

        truth_lines = []
        result_lines = []
        for index, pairs in enumerate(self.result_pairs):
            src, tgt, pred = pairs
            src_tokens, tgt_tokens, pred_tokens = list(src), list(tgt), list(pred)
            if len(src_tokens) != len(tgt_tokens) or len(tgt_tokens) != len(pred_tokens):
                abnormal_pairs.append(pairs)
                continue

            truth_items = []
            result_items = []
            for i in range(len(src_tokens)):
                if src_tokens[i] != tgt_tokens[i]:
                    truth_items.append("%d, %s" % (i + 1, tgt_tokens[i]))

                if src_tokens[i] != pred_tokens[i]:
                    result_items.append("%d, %s" % (i + 1, pred_tokens[i]))

            if len(truth_items) == 0:
                truth_lines.append("A%.5d, 0\n" % index)
            else:
                truth_lines.append("A%.5d, %s\n" % (index, ', '.join(truth_items)))

            if len(result_items) == 0:
                result_lines.append("A%.5d, 0\n" % index)
            else:
                result_lines.append("A%.5d, %s\n" % (index, ', '.join(result_items)))

        mkdir(dir_path)
        path = Path(dir_path)
        with open(path / 'sighan_truth.txt', mode='w', encoding='utf-8') as f:
            f.writelines(truth_lines)

        with open(path / 'sighan_result.txt', mode='w', encoding='utf-8') as f:
            f.writelines(result_lines)

        print("Export SIGHAN result to", dir_path, ". The number of abnormal pairs:", len(abnormal_pairs))


class SighanCSCMetrics:
    """
    SIGHAN 官方的评价指标实现
    """

    def __init__(self):
        # d: detect, c:correct, s:sentence-level
        self.d_tp, self.d_fp, self.d_tn, self.d_fn = 0, 0, 0, 0
        self.c_tp, self.c_fp, self.c_tn, self.c_fn = 0, 0, 0, 0
        self.sd_tp, self.sd_fp, self.sd_tn, self.sd_fn = 0, 0, 0, 0
        self.sc_tp, self.sc_fp, self.sc_tn, self.sc_fn = 0, 0, 0, 0

        self.total_sent_num = 0
        self.abnormal_pairs = []
        self.error_pairs = []

        self.result_pairs = []

    def add_sentence(self, src, tgt, pred):
        self.total_sent_num += 1

        self.result_pairs.append((src, tgt, pred))

        src_tokens, tgt_tokens, pred_tokens = None, None, None
        if type(src) == str and type(tgt) == str and type(pred) == str:
            src = src.replace(" ", "")
            tgt = tgt.replace(" ", "")
            pred = pred.replace(" ", "")
            src_tokens = list(src)
            tgt_tokens = list(tgt)
            pred_tokens = list(pred)

        if not src_tokens:
            self.abnormal_pairs.append((src, tgt, pred))

        if len(src) != len(tgt) or len(tgt) != len(pred):
            self.abnormal_pairs.append((src, tgt, pred))
            return

        if pred != tgt:
            # 预测错了
            self.error_pairs.append((src, tgt, pred))

        self._char_detect_metrics(src_tokens, tgt_tokens, pred_tokens)
        self._char_correct_metrics(src_tokens, tgt_tokens, pred_tokens)
        self._sent_detect_metrics(src_tokens, tgt_tokens, pred_tokens)
        self._sent_correct_metrics(src, tgt, pred)

    def get_results(self):
        char_detect_acc = (self.d_tp + self.d_tn) / (self.d_tp + self.d_fp + self.d_tn + self.d_fn + 1e-8)
        char_detect_p = self.d_tp / (self.d_tp + self.d_fp + 1e-8)
        char_detect_r = self.d_tp / (self.d_tp + self.d_fn + 1e-8)
        char_detect_f1 = (2 * char_detect_p * char_detect_r) / (char_detect_p + char_detect_r + 1e-8)

        char_correct_acc = (self.c_tp + self.c_tn) / (self.c_tp + self.c_fp + self.c_tn + self.c_fn + 1e-8)
        char_correct_p = self.c_tp / (self.c_tp + self.c_fp + 1e-8)
        char_correct_r = self.c_tp / (self.c_tp + self.c_fn + 1e-8)
        char_correct_f1 = (2 * char_correct_p * char_correct_r) / (char_correct_p + char_correct_r + 1e-8)

        sent_detect_acc = (self.sd_tp + self.sd_tn) / (self.sd_tp + self.sd_fp + self.sd_tn + self.sd_fn + 1e-8)
        sent_detect_p = self.sd_tp / (self.sd_tp + self.sd_fp + 1e-8)
        sent_detect_r = self.sd_tp / (self.sd_tp + self.sd_fn + 1e-8)
        sent_detect_f1 = (2 * sent_detect_p * sent_detect_r) / (sent_detect_p + sent_detect_r + 1e-8)

        sent_correct_acc = (self.sc_tp + self.sc_tn) / (self.sc_tp + self.sc_fp + self.sc_tn + self.sc_fn + 1e-8)
        sent_correct_p = self.sc_tp / (self.sc_tp + self.sc_fp + 1e-8)
        sent_correct_r = self.sc_tp / (self.sc_tp + self.sc_fn + 1e-8)
        sent_correct_f1 = (2 * sent_correct_p * sent_correct_r) / (sent_correct_p + sent_correct_r + 1e-8)

        return char_detect_acc, char_detect_p, char_detect_r, char_detect_f1, \
               char_correct_acc, char_correct_p, char_correct_r, char_correct_f1, \
               sent_detect_acc, sent_detect_p, sent_detect_r, sent_detect_f1, \
               sent_correct_acc, sent_correct_p, sent_correct_r, sent_correct_f1

    def print_results(self):
        char_detect_acc, char_detect_p, char_detect_r, char_detect_f1, \
        char_correct_acc, char_correct_p, char_correct_r, char_correct_f1, \
        sent_detect_acc, sent_detect_p, sent_detect_r, sent_detect_f1, \
        sent_correct_acc, sent_correct_p, sent_correct_r, sent_correct_f1 = self.get_results()
        log.info("------------------------------------------------------------")
        log.info("Character-level Detect Acc: %.4f, P: %.4f, R: %.4f, F1: %.4f"
                 % (char_detect_acc, char_detect_p, char_detect_r, char_detect_f1))
        log.info("Character-level Correct Acc: %.4f, P: %.4f, R: %.4f, F1: %.4f"
                 % (char_correct_acc, char_correct_p, char_correct_r, char_correct_f1))
        log.info("Sentence-level Detect Acc: %.4f, P: %.4f, R: %.4f, F1: %.4f"
                 % (sent_detect_acc, sent_detect_p, sent_detect_r, sent_detect_f1))
        log.info("Sentence-level Correct Acc: %.4f, P: %.4f, R: %.4f, F1: %.4f"
                 % (sent_correct_acc, sent_correct_p, sent_correct_r, sent_correct_f1))
        log.info("------------------------------------------------------------")

        log.info("Total Sentences Num: %d, Error Sentences Num: %d" % (self.total_sent_num, len(self.abnormal_pairs)))

    def print_errors(self):
        for src, tgt, pred in self.error_pairs:
            print("---------------------------")
            src_tokens = list(src)
            tgt_tokens = list(tgt)
            pred_tokens = list(pred)
            tgt_detects = [1 if src_tokens[i] != tgt_tokens[i] else 0 for i in range(len(src_tokens))]
            pred_detects = [1 if src_tokens[i] != pred_tokens[i] else 0 for i in range(len(src_tokens))]
            print("src : %s" % src)
            print("tgt : %s" % render_color_for_text(tgt, tgt_detects, 'green'))
            print("pred: %s" % render_color_for_text(pred, pred_detects, 'red'))

    def print_abnormal_pairs(self):
        for src, tgt, pred in self.abnormal_pairs:
            print("*********************************")
            print("src : %s" % src)
            print("tgt : %s" % tgt)
            print("pred: %s" % pred)

    def _char_detect_metrics(self, src_tokens, tgt_tokens, pred_tokens):
        for src, tgt, pred in zip(src_tokens, tgt_tokens, pred_tokens):
            # 该纠的字，纠了，纠没纠对不管
            if src != tgt and pred != src:
                self.d_tp += 1
            # 该纠的字，没纠
            elif src != tgt and pred == src:
                self.d_fn += 1
            # 不该纠的字，纠了
            elif src == tgt and pred != src:
                self.d_fp += 1
            # 不该纠的字，没纠
            elif src == tgt and pred == src:
                self.d_tn += 1
            else:
                raise Exception("Code Bug???")

    def _char_correct_metrics(self, src_tokens, tgt_tokens, pred_tokens):
        for src, tgt, pred in zip(src_tokens, tgt_tokens, pred_tokens):
            # 该纠的字，纠了也纠对了
            if src != tgt and pred != src and pred == tgt:
                self.c_tp += 1
            # 该纠的字，没纠或没纠对
            elif src != tgt and pred != tgt:
                self.c_fn += 1
            # 不该纠的字，纠了
            elif src == tgt and pred != src:
                self.c_fp += 1
            # 不该纠的字，没纠
            elif src == tgt and pred == src:
                self.c_tn += 1
            else:
                raise Exception("Code Bug???")

    def _sent_detect_metrics(self, src_tokens, tgt_tokens, pred_tokens):
        tgt_detects = [1 if src_tokens[i] != tgt_tokens[i] else 0 for i in range(len(src_tokens))]
        pred_detects = [1 if src_tokens[i] != pred_tokens[i] else 0 for i in range(len(src_tokens))]

        # 该纠，且该纠的字，都纠了，纠没纠对不管；不该纠的字，都没纠
        if sum(tgt_detects) > 0 and tgt_detects == pred_detects:
            self.sd_tp += 1
        # 该纠，但未纠或把不该纠的字纠了
        elif sum(tgt_detects) > 0 and tgt_detects != pred_detects:
            self.sd_fn += 1
        # 不该纠，但纠了
        elif sum(tgt_detects) == 0 and sum(pred_detects) > 0:
            self.sd_fp += 1
        # 不该纠，没纠
        elif sum(tgt_detects) == 0 and sum(pred_detects) == 0:
            self.sd_tn += 1
        else:
            raise Exception("Code Bug???")

    def _sent_correct_metrics(self, src, tgt, pred):
        # 该纠，且纠对了
        if src != tgt and pred != src and pred == tgt:
            self.sc_tp += 1
        # 该纠，未纠或纠错了
        elif src != tgt and pred != tgt:
            self.sc_fn += 1
        # 不该纠，但纠了
        elif src == tgt and pred != src:
            self.sc_fp += 1
        # 不该纠，未纠
        elif src == tgt and pred == src:
            self.sc_tn += 1
        else:
            raise Exception("Code Bug???")

    def export_sigan_format(self, dir_path='outputs'):
        abnormal_pairs = []

        truth_lines = []
        result_lines = []
        for index, pairs in enumerate(self.result_pairs):
            src, tgt, pred = pairs
            src_tokens, tgt_tokens, pred_tokens = list(src), list(tgt), list(pred)
            if len(src_tokens) != len(tgt_tokens) or len(tgt_tokens) != len(pred_tokens):
                abnormal_pairs.append(pairs)
                continue

            truth_items = []
            result_items = []
            for i in range(len(src_tokens)):
                if src_tokens[i] != tgt_tokens[i]:
                    truth_items.append("%d, %s" % (i + 1, tgt_tokens[i]))

                if src_tokens[i] != pred_tokens[i]:
                    result_items.append("%d, %s" % (i + 1, pred_tokens[i]))

            if len(truth_items) == 0:
                truth_lines.append("A%.5d, 0\n" % index)
            else:
                truth_lines.append("A%.5d, %s\n" % (index, ', '.join(truth_items)))

            if len(result_items) == 0:
                result_lines.append("A%.5d, 0\n" % index)
            else:
                result_lines.append("A%.5d, %s\n" % (index, ', '.join(result_items)))

        mkdir(dir_path)
        path = Path(dir_path)
        with open(path / 'sighan_truth.txt', mode='w', encoding='utf-8') as f:
            f.writelines(truth_lines)

        with open(path / 'sighan_result.txt', mode='w', encoding='utf-8') as f:
            f.writelines(result_lines)

        print("Export SIGHAN result to", dir_path, ". The number of abnormal pairs:", len(abnormal_pairs))
