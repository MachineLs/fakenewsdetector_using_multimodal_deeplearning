from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import interp
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, average_precision_score
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import OneHotEncoder


def metrics(truth, pred, prob, file_path):
    truth = [i.cpu().numpy() for i in truth]
    pred = [i.cpu().numpy() for i in pred]
    prob = [i.cpu().numpy() for i in prob]

    pred = np.concatenate(pred, axis=0)
    truth = np.concatenate(truth, axis=0)
    prob = np.concatenate(prob, axis=0)
    prob = prob[:, 1]

    f_score_micro = f1_score(truth, pred, average='micro', zero_division=0)
    f_score_macro = f1_score(truth, pred, average='macro', zero_division=0)
    f_score_weighted = f1_score(truth, pred, average='weighted', zero_division=0)
    accuarcy = accuracy_score(truth, pred)

    s = ''
    print('accuracy', accuarcy)
    s += 'accuracy' + str(accuarcy) + '\n'
    print('f_score_micro', f_score_micro)
    s += 'f_score_micro' + str(f_score_micro) + '\n'
    print('f_score_macro', f_score_macro)
    s += 'f_score_macro' + str(f_score_macro) + '\n'
    print('f_score_weighted', f_score_weighted)
    s += 'f_score_weighted' + str(f_score_weighted) + '\n'

    fpr, tpr, thresholds = roc_curve(truth, prob)
    AUC = auc(fpr, tpr)
    print('AUC', AUC)
    s += 'AUC' + str(AUC) + '\n'
    df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
    df.to_csv(file_path)

    return s


def report_per_class(truth, pred):
    truth = [i.cpu().numpy() for i in truth]
    pred = [i.cpu().numpy() for i in pred]

    pred = np.concatenate(pred, axis=0)
    truth = np.concatenate(truth, axis=0)

    report = classification_report(truth, pred, zero_division=0, output_dict=True)

    s = ''
    class_labels = [k for k in report.keys() if k not in ['micro avg', 'macro avg', 'weighted avg', 'samples avg']]
    for class_label in class_labels:
        print('class_label', class_label)
        s += 'class_label' + str(class_label) + '\n'
        s += str(report[class_label])
        print(report[class_label])

    return s


def multiclass_acc(truth, pred):
    truth = [i.cpu().numpy() for i in truth]
    pred = [i.cpu().numpy() for i in pred]

    pred = np.concatenate(pred, axis=0)
    truth = np.concatenate(truth, axis=0)

    return accuracy_score(truth, pred)


def roc_auc_plot(truth, score, num_class=2, fname='roc.png'):
    truth = [i.cpu().numpy() for i in truth]
    score = [i.cpu().numpy() for i in score]

    truth = np.concatenate(truth, axis=0)
    score = np.concatenate(score, axis=0)

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(truth.reshape(-1, 1))
    label_onehot = enc.transform(truth.reshape(-1, 1)).toarray()

    fpr_dict = dict()
    tpr_dict = dict()
    roc_auc_dict = dict()
    for i in range(num_class):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(label_onehot[:, i], score[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
    # micro
    fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(label_onehot.ravel(), score.ravel())
    roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])

    # macro
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(num_class)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_class):
        mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= num_class
    fpr_dict["macro"] = all_fpr
    tpr_dict["macro"] = mean_tpr
    roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])

    plt.figure()

    lw = 2
    plt.plot(fpr_dict["micro"], tpr_dict["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc_dict["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr_dict["macro"], tpr_dict["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc_dict["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(num_class), colors):
        plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc_dict[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(fname)
    # plt.show()


def precision_recall_plot(truth, score, num_class=2, fname='pr.png'):
    truth = [i.cpu().numpy() for i in truth]
    score = [i.cpu().numpy() for i in score]

    truth = np.concatenate(truth, axis=0)
    score = np.concatenate(score, axis=0)

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(truth.reshape(-1, 1))
    label_onehot = enc.transform(truth.reshape(-1, 1)).toarray()

    # Call the Sklearn library, calculate the precision and recall corresponding to each category
    precision_dict = dict()
    recall_dict = dict()
    average_precision_dict = dict()
    for i in range(num_class):
        precision_dict[i], recall_dict[i], _ = precision_recall_curve(label_onehot[:, i], score[:, i])
        average_precision_dict[i] = average_precision_score(label_onehot[:, i], score[:, i])
        print(precision_dict[i].shape, recall_dict[i].shape, average_precision_dict[i])

    # micro
    precision_dict["micro"], recall_dict["micro"], _ = precision_recall_curve(label_onehot.ravel(),
                                                                              score.ravel())
    average_precision_dict["micro"] = average_precision_score(label_onehot, score, average="micro")

    # macro
    all_fpr = np.unique(np.concatenate([precision_dict[i] for i in range(num_class)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_class):
        mean_tpr += interp(all_fpr, precision_dict[i], recall_dict[i])
    mean_tpr /= num_class
    precision_dict["macro"] = all_fpr
    recall_dict["macro"] = mean_tpr
    average_precision_dict["macro"] = auc(precision_dict["macro"], recall_dict["macro"])

    plt.figure()
    plt.subplots(figsize=(16, 10))
    lw = 2
    plt.plot(precision_dict["micro"], recall_dict["micro"],
             label='micro-average Precision-Recall curve (area = {0:0.2f})'
                   ''.format(average_precision_dict["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(precision_dict["macro"], recall_dict["macro"],
             label='macro-average Precision-Recall curve (area = {0:0.2f})'
                   ''.format(average_precision_dict["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(num_class), colors):
        plt.plot(precision_dict[i], recall_dict[i], color=color, lw=lw,
                 label='Precision-Recall curve of class {0} (area = {1:0.2f})'
                       ''.format(i, average_precision_dict[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])

    plt.legend(loc="lower left")
    plt.savefig(fname=fname)
    # plt.show()


def saving_in_tensorboard(config, x, y, fname='embedding'):
    x = [i.cpu().numpy() for i in x]
    y = [i.cpu().numpy() for i in y]

    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    z = pd.DataFrame(y)[0].apply(lambda i: config.classes[i]).values

    # config.writer.add_embedding(mat=x, label_img=y, metadata=z, tag=fname)

def plot_tsne(config, x, y, fname='tsne.png'):
    x = [i.cpu().numpy() for i in x]
    y = [i.cpu().numpy() for i in y]

    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)

    y = pd.DataFrame(y)[0].apply(lambda i: config.classes[i]).values

    tsne = TSNE(n_components=2, verbose=1, init="pca", perplexity=10, learning_rate=1000)
    tsne_proj = tsne.fit_transform(x)

    fig, ax = plt.subplots(figsize=(16, 10))

    palette = sns.color_palette("bright", 2)
    sns.scatterplot(tsne_proj[:, 0], tsne_proj[:, 1], hue=y, legend='full', palette=palette)

    ax.legend(fontsize='large', markerscale=2)
    plt.title('tsne of ' + str(fname.split('/')[-1].split('.')[0]))
    plt.savefig(fname=fname)
    # plt.show()


def plot_pca(config, x, y, fname='pca.png'):
    x = [i.cpu().numpy() for i in x]
    y = [i.cpu().numpy() for i in y]

    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)

    y = pd.DataFrame(y)[0].apply(lambda i: config.classes[i]).values

    pca = PCA(n_components=2)
    pca_proj = pca.fit_transform(x)

    fig, ax = plt.subplots(figsize=(16, 10))

    palette = sns.color_palette("bright", 2)
    sns.scatterplot(pca_proj[:, 0], pca_proj[:, 1], hue=y, legend='full', palette=palette)

    ax.legend(fontsize='large', markerscale=2)
    plt.title('pca of ' + str(fname.split('/')[-1].split('.')[0]))
    plt.savefig(fname=fname)
    # plt.show()
