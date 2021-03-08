from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def get_results_wo_clf(feature_vec, true_labels, roc=True, conf_matrix=True, fpr_rate=False, thd=False):
    # normalize 0-1
    # pred_probs = (feature_vec - np.min(feature_vec))/(np.max(feature_vec)-np.min(feature_vec))
    pred_probs = feature_vec
    fpr, tpr, thresholds = roc_curve(true_labels, pred_probs, pos_label=1)
    optimal_idx = np.argmax(tpr - fpr)

    optimal_threshold = thresholds[optimal_idx]
    pred_labels = np.zeros_like(feature_vec)
    pred_labels[pred_probs >= optimal_threshold] = 1
    if fpr_rate:
        optimal_idx = np.argmin(np.abs(fpr - fpr_rate))
        optimal_idx = np.where(fpr[optimal_idx] == fpr)[0][-1]
        # optimal_idx = np.argmax(tpr - fpr_rate)
        optimal_threshold = thresholds[optimal_idx]
        pred_labels = np.zeros_like(feature_vec)
        pred_labels[pred_probs >= optimal_threshold] = 1
        accuracy = pred_labels.sum() / len(pred_labels)
    if thd:
        optimal_idx = np.argmin(np.abs(thd - thresholds))
        optimal_threshold = thresholds[optimal_idx]
        pred_labels = np.zeros_like(feature_vec)
        pred_labels[pred_probs >= optimal_threshold] = 1
        accuracy = pred_labels.sum() / len(pred_labels)

    print('AUC = ', auc(fpr, tpr) * 100)
    print('FPR = ', fpr[optimal_idx] * 100)
    print('TPR = ', tpr[optimal_idx] * 100)
    print('Optimal threshold = ', optimal_threshold)
    # print('ACCURACY = ', accuracy*100)

    if roc:
        a = plt.figure(1)
        plt.plot(fpr, tpr, color='orange', label='ROC')
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        # plt.show()

    if conf_matrix:
        b = plt.figure(2)
        Conf = sns.heatmap(confusion_matrix(true_labels, pred_labels), annot=True, cmap='BuPu', cbar=False, fmt='g')
        plt.show()
    return pred_labels


def get_results(pred_labels, pred_probs, true_labels, roc=True, conf_matrix=True, fpr_rate=False):
    """
    :param probability_pred: normalized probability vector
    :param true_label: ground truth labels
    :param roc: True for creating roc plot
    :param conf_matrix: True for creating roc plot
    :param fpr_rate: a number between 0-1 if a specific fpr rate is needed
    :return: optimal prediction vector [0/1]
    """

    fpr, tpr, thresholds = roc_curve(true_labels, pred_probs, pos_label=1)
    optimal_idx = np.argmax(tpr - fpr)
    # if fpr_rate:
    #     optimal_idx = np.argmax(fpr - fpr_rate)
    #     optimal_threshold = thresholds[optimal_idx]
    #     pred_labels = []
    #     for pred_prob in pred_probs:
    #         if pred_prob >= optimal_threshold:
    #             pred_labels.append(1)
    #         else:
    #             pred_labels.append(0)
    #
    #     pred_labels = np.asarray(pred_labels)
    #
    # accuracy = pred_labels.sum() / len(pred_labels)
    optimal_threshold = thresholds[optimal_idx]

    print('AUC = ', auc(fpr, tpr) * 100)
    print('FPR = ', fpr[optimal_idx] * 100)
    print('TPR = ', tpr[optimal_idx] * 100)
    # print('ACCURACY = ', accuracy*100)

    if roc:
        a = plt.figure(1)
        plt.plot(fpr, tpr, color='orange', label='ROC')
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        # plt.show()

    if conf_matrix:
        b = plt.figure(2)
        Conf = sns.heatmap(confusion_matrix(true_labels, pred_labels), annot=True, cmap='BuPu', cbar=False, fmt='g')
        plt.show()
    return pred_labels


class Results:
    def __init__(self, y_pred, y_true):
        # Initialize class objects
        # y_pred is the cosine similarity vector
        if (type(y_pred) == 'pandas.core.frame.DataFrame'):
            y_pred.to_numpy()

        self.y_pred = np.asarray(y_pred)
        self.y_true = np.asarray(y_true)
        self.y_true[self.y_true > 0] = 1

    def Model_check(self):
        # Plot "histogram" senity check
        ax = plt.subplot(111)
        plt.scatter(self.y_pred, self.y_true)
        plt.show()
