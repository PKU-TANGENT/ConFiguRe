import numpy as np


# Metric for end2end model. We adopt exact match, which requires start index, end index and fig type should match.
class MetricForRecognitionCRF:
    def __init__(self, id2lb, lb2id, logger):
        # this is for fig level
        self.gold_fig_num = np.zeros(len(id2lb))
        self.pred_fig_num = np.zeros(len(id2lb))
        self.correct_fig_num = np.zeros(len(id2lb))

        self.overpred_fig_num = np.zeros(len(id2lb))
        self.subpred_fig_num = np.zeros(len(id2lb))
        self.lappred_fig_num = np.zeros(len(id2lb))
        self.mispred_fig_num = np.zeros(len(id2lb))
        self.wrgpred_fig_num = np.zeros(len(id2lb))

        self.id2lb = id2lb
        self.lb2id = lb2id
        self.logger = logger

    def add(self, out, labels):
        # add gold fig
        gold_fig_list = self.get_fig_list(labels)
        for fig in gold_fig_list:
            self.gold_fig_num[fig[2]] += 1

        # add pred fig
        pred_fig_list = self.get_fig_list(out)
        for fig in pred_fig_list:
            self.pred_fig_num[fig[2]] += 1

        # add correct num
        correct_fig_list = [x for x in gold_fig_list if x in pred_fig_list]
        for fig in correct_fig_list:
            self.correct_fig_num[fig[2]] += 1

        for pred in pred_fig_list:
            pred_rg = set(range(pred[0], pred[1]+1))
            wrong_flag = 1
            for gold in gold_fig_list:
                gold_rg = set(range(gold[0], gold[1]+1)) 

                # check if it is over pred (i.e. predicting too much clauses)
                if pred[2] == gold[2] and pred_rg.issuperset(gold_rg) :
                    self.overpred_fig_num[gold[2]] += 1

                # check if it is sub pred (i.e. predicting too few clauses)
                if pred[2] == gold[2] and pred_rg.issubset(gold_rg) :
                    self.subpred_fig_num[gold[2]] += 1

                # check if it is lap pred (i.e. predicting wrong clauses but do overlaps)
                if pred[2] == gold[2] and pred_rg.union(gold_rg) < pred_rg:
                    self.subpred_fig_num[gold[2]] += 1

                # check if it is mis pred (i.e. predicting right range but wrong figure type)
                if pred[0] == gold[0] and pred[1] == gold[1] and pred[2] != gold[2]:
                    self.mispred_fig_num[gold[2]] += 1

                # check if it is wrong pred (i.e. predicting completely wrong clauses)
                if len(pred_rg.union(gold_rg)) > 0:
                    wrong_flag = 0

            if wrong_flag:
                self.wrgpred_fig_num[pred[2]] += 1


        return

    def get_fig_list(self, pred_list):
        """
        returns:
            fig_list: a list of tuple (st, ed, fig_type), where both st and ed is inclusive
        """
        pred_list_len = len(pred_list)
        prev_idx, prev = 0, self.lb2id["O"]  # 0 stands for label "O"
        cur_idx = 0
        fig_list = list()
        while cur_idx < pred_list_len:
            while cur_idx < pred_list_len - 1 and not self.id2lb[pred_list[cur_idx]].startswith("B-"):
                cur_idx += 1
            prev_idx = cur_idx # prev_idx = pred_list_len - 1 or starts with B
            cur_idx +=1
            while cur_idx < pred_list_len and self.id2lb[pred_list[cur_idx]]=="I" + self.id2lb[pred_list[prev_idx]][1:]:
                cur_idx += 1
            if not self.id2lb[pred_list[prev_idx]].startswith("B-"):
                break
            fig_list.append((prev_idx, cur_idx-1, pred_list[prev_idx]))

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
        f1_list = list()
        for idx, fig_type in enumerate(self.id2lb):
            if fig_type == "O" or fig_type == "<start_token>" or fig_type == "<end_token>":
                continue
            precis, recall, f1_score = self.helper_get_prf(
                self.correct_fig_num[idx], self.pred_fig_num[idx], self.gold_fig_num[idx])
            self.logger.info('{} | Num:{:5.0f} | P: {:5.2f} | R: {:5.2f} | F1: {:5.2f}'.format(
                fig_type, self.gold_fig_num[idx], precis * 100, recall * 100, f1_score * 100))
            f1_list.append(f1_score)

        if show_overall is True:
            correct_num = sum(self.correct_fig_num)
            pred_num = sum(self.pred_fig_num)
            gold_num = sum(self.gold_fig_num)

            precis, recall, f1_score = self.helper_get_prf(
                correct_num, pred_num, gold_num)
            self.logger.info('Overall performance | Total num:{:5.0f} | P: {:5.2f} | R: {:5.2f} | Micro F1: {:5.2f}  | Macro F1: {:5.2f}'.format(
                gold_num, precis * 100, recall * 100, f1_score * 100, sum(f1_list)/len(f1_list) * 200))

            return f1_score

        return
