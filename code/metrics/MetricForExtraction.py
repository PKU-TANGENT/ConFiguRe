import numpy as np


# Metric for end2end model. We adopt exact match, which requires start index, end index and fig type should match.
class MetricForExtraction:
    def __init__(self, id2lb, lb2id, logger):
        # this is for fig level
        self.gold_fig_num = 0
        self.pred_fig_num = 0
        self.correct_fig_num = 0

        self.id2lb = id2lb
        self.lb2id = lb2id
        self.logger = logger

    def add(self, out, labels):
        # add gold fig
        gold_fig_list = self.get_fig_list(labels)
        for _ in gold_fig_list:
            self.gold_fig_num += 1

        # add pred fig
        pred_fig_list = self.get_fig_list(out)
        for _ in pred_fig_list:
            self.pred_fig_num += 1

        # add correct num
        correct_fig_list = [x for x in gold_fig_list if x in pred_fig_list]
        for _ in correct_fig_list:
            self.correct_fig_num += 1

        return

    def get_fig_list(self, pred_list):
        """
        returns:
            fig_list: a list of tuple (st, ed), where both st and ed is inclusive
        """
        pred_list_len = len(pred_list)
        # prev_idx, prev = 0, self.lb2id["O"] 
        flag, st_idx = 0, -1
        fig_list = list()
        for i in range(pred_list_len):

            if flag == 1:
                if self.id2lb[pred_list[i]] ==  "I":
                    if i == pred_list_len - 1:
                        fig_list.append((st_idx, i )) 
                    continue
                else:
                    fig_list.append((st_idx, i-1))
                    flag = 0

            if self.id2lb[pred_list[i]] == "B":
                flag = 1
                st_idx = i
                if i == pred_list_len - 1:
                    fig_list.append((i, i ))

        return fig_list

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

        if show_overall is True:
            correct_num = self.correct_fig_num
            pred_num = self.pred_fig_num
            gold_num = self.gold_fig_num

            precis, recall, f1_score = self.helper_get_prf(
                correct_num, pred_num, gold_num)
            self.logger.info('Overall performance | Total num:{:5.0f} | P: {:5.2f} | R: {:5.2f} | F1: {:5.2f}'.format(
                gold_num, precis * 100, recall * 100, f1_score * 100))

            return f1_score

        return
