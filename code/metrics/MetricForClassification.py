import numpy as np


# Metric for end2end model. We adopt exact match, which requires start index, end index and fig type should match.
class MetricForClassification:
    def __init__(self, id2lb, lb2id, logger):
        # this is for fig level
        self.gold_fig_num = np.zeros(len(id2lb))
        self.pred_fig_num = np.zeros(len(id2lb))
        self.correct_fig_num = np.zeros(len(id2lb))
        self.confusion_matrix = np.zeros((len(id2lb), len(id2lb)), dtype="float32")

        self.id2lb = id2lb
        self.lb2id = lb2id
        self.logger = logger

    def add(self, pred_list, label_list):
        for idx in range(len(label_list)):
            label = label_list[idx]
            pred = pred_list[idx]

            # add gold fig
            self.gold_fig_num[label] += 1

            # add pred fig
            self.pred_fig_num[pred] += 1

            # add correct num
            self.correct_fig_num[pred] += (label == pred)

            self.confusion_matrix[label][pred] += 1

        return

    def helper_get_prf(self, correct_num, pred_num, gold_num):
        if pred_num == 0 or gold_num == 0:
            return 0, 0, 0
        precis = correct_num / pred_num
        recall = correct_num / gold_num
        if precis + recall == 0:
            return 0, 0, 0
        f1_score = 2 * precis * recall / (precis + recall)
        return precis, recall, f1_score

    def get_metrics(self, show_overall):

        f1_list = np.zeros(len(self.id2lb))
        for idx, fig_type in enumerate(self.id2lb):
            if fig_type == "O":
                continue
            precis, recall, f1_score = self.helper_get_prf(
                self.correct_fig_num[idx], self.pred_fig_num[idx], self.gold_fig_num[idx])
            self.logger.info('{} | Num:{:5.0f} | P: {:5.2f} | R: {:5.2f} | F1: {:5.2f}'.format(
                fig_type, self.gold_fig_num[idx], precis * 100, recall * 100, f1_score * 100))
            f1_list[idx] = f1_score

        if show_overall is True:
            correct_num = sum(self.correct_fig_num)
            pred_num = sum(self.pred_fig_num)
            gold_num = sum(self.gold_fig_num)

            precis, recall, f1_score = self.helper_get_prf(
                correct_num, pred_num, gold_num)
            self.logger.info('Overall performance | Total num:{:5.0f} | P: {:5.2f} | R: {:5.2f} | Micro F1: {:5.2f} | Macro F1: {:5.2f}'.format(
                gold_num, precis * 100, recall * 100, f1_score * 100, np.average(f1_list) * 100))

            return f1_score

    def save_confusion_matrix(self, epoch):

        # normalize
        row_sum = self.confusion_matrix.sum(axis=1)
        for i in range(len(self.id2lb)):
            self.confusion_matrix[i] = self.confusion_matrix[i] / row_sum[i]
         
        # write
        with open(f"./Vectors/confusion_epoch{epoch}.txt",'w') as fout:
            np.savetxt(fout, self.confusion_matrix, fmt="%.3f")

        return
