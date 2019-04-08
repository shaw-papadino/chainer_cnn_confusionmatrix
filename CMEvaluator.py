import chainer
from chainer import reporter as reporter_module
from chainer.training import extensions
from chainer import function
import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score

class CMEvaluator(extensions.Evaluator):
    """
    Confusion Matrixで再現率、適合率を算出するEvaluator
    """
    default_name="mycm"
    def evaluate(self):
        # iterator, modelを設定
        iterator = self._iterators['main']
        model = self._targets['main']
        # 別で行う関数があるなら設定
        eval_func = self.eval_func or model

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        recall_count = 0
        accuracy_count = 0
        lcount = 0
        for i, batch in enumerate(it):
            observation = {}
            with reporter_module.report_scope(observation):
                in_arrays = self.converter(batch, self.device)
                with function.no_backprop_mode():
                    if isinstance(in_arrays, tuple):
                        eval_func(*in_arrays)
                        re, ac = self.cm(in_arrays)
                    elif isinstance(in_arrays, dict):
                        eval_func(**in_arrays)
                        re, ac = self.cm(in_arrays)
                    else:
                        eval_func(in_arrays)
                        re, ac = self.cm(in_arrays)
                    recall_count = recall_count + re
                    accuracy_count = accuracy_count + ac

            summary.add(observation)
            lcount = i

        cm_observation = {}

        cm_observation["cmrecall"] = round(recall_count / (lcount + 1) ,3)
        cm_observation["cmaccuracy"] = round(accuracy_count / (lcount + 1), 3)
        # print(cm_observation)
        # print(summary.compute_mean())
        summary.add(cm_observation)

        return summary.compute_mean()




    def cm(self, in_arrays):
        model = self._targets['main']

        _, labels = in_arrays
        if self.device >= 0:
            labels = chainer.cuda.to_cpu(labels)

        y = model.y.data
        if self.device >= 0:
            y = chainer.cuda.to_cpu(y)
        y = y.argmax(axis=1)
        cmatrix = np.zeros((2,2))
        cmatrix = np.array(confusion_matrix(labels,y))
        # print(cmatrix)
        recall = round(cmatrix[0,0] / (cmatrix[0,0] + cmatrix[0,1]), 3)
        accuracy = round(cmatrix[0,0] / (cmatrix[0,0] + cmatrix[1,0]),3)

        return recall, accuracy
