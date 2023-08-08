import tqdm


def render_color_for_text(text, indices, color='red', format='console'):
    color_indices = {
        'black': '30',
        'red': '31',
        'green': '32',
        'yellow': '33'
    }

    text = text.replace(" ", "")
    char_list = list(text)
    for i in range(len(indices)):
        if indices[i]:
            if format in ['console', "sh", "shell"]:
                char_list[i] = "\033[" + color_indices.get(color, '30') + "m" + text[i] + "\033[0m"
            elif format in ['markdown', "md"]:
                char_list[i] = ":%s[%s]" % (color, char_list[i])

    return ''.join(char_list)


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
        print("------------------------------------------------------------")
        print("Sentence-level Detect Acc: %.4f, P: %.4f, R: %.4f, F1: %.4f"
              % self._get_sent_level_detect_metrics())
        print("Sentence-level Correct Acc: %.4f, P: %.4f, R: %.4f, F1: %.4f"
              % self._get_sent_level_correct_metrics())
        print("------------------------------------------------------------")

        print("Total Sentences Num: %d, Error Sentences Num: %d" % (self.total_sent_num, len(self.abnormal_pairs)))

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


def eval_model(predict_func, test_data_path):
    csc_metrics = CSCMetrics()

    with open(test_data_path, encoding='utf-8') as f:
        lines = f.readlines()[1:]

    for line in tqdm.tqdm(lines):
        src, tgt = line.split(",")
        src = src.strip()
        tgt = tgt.strip()

        pred = predict_func(src)
        csc_metrics.add_sentence(src, tgt, pred)

    csc_metrics.print_results()
    csc_metrics.print_errors()
    csc_metrics.print_abnormal_pairs()




