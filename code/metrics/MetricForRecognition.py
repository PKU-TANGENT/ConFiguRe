import numpy as np


# Metric for end2end model. We adopt exact match, which requires start index, end index and fig type should match.
class MetricForRecognition:
    def __init__(self, args):
        # this is for fig level
        self.gold_fig_num = np.zeros(len(args.figs))
        self.pred_fig_num = np.zeros(len(args.figs))
        self.correct_fig_num = np.zeros(len(args.figs))

        self.id2lb = args.id2lb
        self.lb2id = args.lb2id
        self.figs = args.figs
        self.logger = args.logger

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

        return

    def get_fig_list(self, pred_list):
        """
        returns:
            fig_list: a list of tuple (st, ed, fig_type), where both st and ed is inclusive
        """
        pred_list_len = len(pred_list)
        # prev_idx, prev = 0, self.lb2id["O"] 
        flag, st_idx = 0, -1
        fig_list = list()
        for i in range(pred_list_len):

            if flag == 1:
                if self.id2lb[pred_list[i]] ==  "I" + self.id2lb[pred_list[st_idx]][1:]:
                    if i == pred_list_len - 1:
                        fig_list.append((st_idx, i, self.figs.index(self.id2lb[pred_list[i]][2:]) )) 
                    continue
                else:
                    fig_list.append((st_idx, i-1, self.figs.index(self.id2lb[pred_list[i-1]][2:])))
                    flag = 0

            if self.id2lb[pred_list[i]].startswith("B-"):
                flag = 1
                st_idx = i
                if i == pred_list_len - 1:
                    fig_list.append((i, i, self.figs.index(self.id2lb[pred_list[i]][2:])))

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

        f1_list = np.zeros(len(self.figs))
        for idx, fig_type in enumerate(self.figs):

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

        return
