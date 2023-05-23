import numpy as np
import matplotlib.pyplot as plt


def get_p_distribution(y_true_label, y_test_pred, savepath):
    # save_path = 'result/R2_feature_256_block_1_test_seed_0507_ratio_04_01'
    # savepath = 'result/R2_feature_256_block_1_test_seed_0507_ratio_04_01/matrix.png'
    # os.makedirs(save_path, exist_ok=True)
    TP = []
    FN = []
    FP = []
    TN = []
    for true_label, y_pred in zip(y_true_label, y_test_pred):
        pred_label = np.argmax(y_pred)

        if true_label == 1 and pred_label == 1:
            TP.append(y_pred)
        if true_label == 1 and pred_label == 0:
            FN.append(y_pred)
        if true_label == 0 and pred_label == 1:
            FP.append(y_pred)
        if true_label == 0 and pred_label == 0:
            TN.append(y_pred)

    TP = np.array(TP)
    TN = np.array(TN)
    FP = np.array(FP)
    FN = np.array(FN)

    fig = plt.figure(figsize=[10, 6])
    ide_style = ['TP', 'FP', 'FN', 'TN']

    for i, item in enumerate([TP, FP, FN, TN]):
        ax1 = fig.add_subplot(2, 2, i + 1)
        ax1.hist(y_test_pred[:, 1], 38)
        ax1.tick_params(top=True, right=True)
        if item.shape[0] > 0:
            if ide_style[i] == 'TP':
                data = item[:, 1]
            if ide_style[i] == 'FP':
                data = item[:, 1]
            if ide_style[i] == 'FN':
                data = item[:, 1]
            if ide_style[i] == 'TN':
                data = item[:, 1]
            ax1.hist(data, 20, label='%s\nnum=%d' % (ide_style[i], data.shape[0]))
        else:
            ax1.hist(y_test_pred[:, 1], 38, label='%s\nnum=%d' % (ide_style[i], 0))
        ax1.legend()
    fig.savefig(savepath, dpi=600, format='png')
    plt.close(fig)


def plot_train_result(Result, savepath=None):
    # colnames = ['acc_kmeans', 'recall_train', 'precision_train', 'f1_train', 'loss_value_train', 'accuracy_train',
    #             'recall_test', 'precision_test', 'f1_test', 'loss_value_test', 'accuracy_test']
    acc_kmeans = Result['acc_kmeans'].values

    recall_train = Result['recall_train'].values
    precision_train = Result['precision_train'].values
    f1_train = Result['f1_train'].values
    loss_value_train = Result['loss_value_train'].values
    accuracy_train = Result['accuracy_train'].values

    recall_test = Result['recall_test'].values
    precision_test = Result['precision_test'].values
    f1_test = Result['f1_test'].values
    loss_value_test = Result['loss_value_test'].values
    accuracy_test = Result['accuracy_test'].values

    # 画损失函数图
    fig = plt.figure(figsize=(10, 6))
    # 坐标轴的刻度设置向内(in)或向外(out)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    ax1 = fig.add_subplot(221)
    ax1.tick_params(top=True, right=True)
    ax1.plot(loss_value_train, label='Train loss')
    ax1.plot(loss_value_test, label='Test loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # subplot acc_train
    ax3 = fig.add_subplot(222)
    ax3.tick_params(top=True, right=True)
    ax3.plot(accuracy_train, label='Train accuracy')
    ax3.plot(accuracy_test, label='Test accuracy')
    ax3.plot(acc_kmeans, label='K-Means accuracy')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Accuracy')
    ax3.legend()

    # subplot acc_train
    ax2 = fig.add_subplot(223)
    ax2.tick_params(top=True, right=True)
    ax2.plot(recall_train, label='Train recall')
    ax2.plot(recall_test, label='Test recall')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Recall')
    ax2.legend()

    # subplot acc_train
    ax4 = fig.add_subplot(224)
    ax4.tick_params(top=True, right=True)
    ax4.plot(precision_train, label='Train precision')
    ax4.plot(precision_test, label='Test precision')
    ax4.plot(f1_train, label='Train f1')
    ax4.plot(f1_test, label='Test f1')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Precision & f1')
    ax4.legend()

    plt.tight_layout()
    if savepath is None:
        plt.show()
    else:
        fig.savefig(savepath, dpi=600, format='png')
        plt.close(fig)


def plot_test_acc_nmi(Result, savepath=None):
    # colnames = ['acc_kmeans', 'recall_train', 'precision_train', 'f1_train', 'loss_value_train', 'accuracy_train',
    #             'recall_test', 'precision_test', 'f1_test', 'loss_value_test', 'accuracy_test', 'NMI_t_t1']
    recall_test = Result['recall_test'].values
    precision_test = Result['precision_test'].values
    f1_test = Result['f1_test'].values
    loss_value_test = Result['loss_value_test'].values
    accuracy_test = Result['accuracy_test'].values
    NMI_t_label = Result['NMI_t_label'].values
    NMI_t_t1 = Result['NMI_t_t1'].values
    resign_num_ratio = Result['cluster_resign_num'].values / Result['cluster_resign_num'].values.max()

    # 画损失函数图
    fig = plt.figure(figsize=(10, 3))
    # 坐标轴的刻度设置向内(in)或向外(out)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    ax1 = fig.add_subplot(131)
    ax1.tick_params(top=True, right=True)
    ax1.plot(NMI_t_label, label='MNI t / labels')
    ax1.plot(NMI_t_t1, label='NMI t-1 / t')
    ax1.plot(resign_num_ratio, label='Reassignment ratio')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('MNI Test Dataset')
    ax1.legend()

    # subplot acc_train
    ax2 = fig.add_subplot(132)
    ax2.tick_params(top=True, right=True)
    ax2.plot(recall_test, label='Test recall')
    ax2.plot(precision_test, label='Test precision')

    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Evaluation Indicators')
    ax2.legend()

    ax3 = fig.add_subplot(133)
    ax3.tick_params(top=True, right=True)
    ax3.plot(f1_test, label='Test F1 score')
    ax3.plot(accuracy_test, label='Test accuracy')

    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Evaluation Indicators')
    ax3.legend()

    plt.tight_layout()
    if savepath is None:
        plt.show()
    else:
        fig.savefig(savepath, dpi=600, format='png')
        plt.close(fig)


def plot_test_acc_reassign(Result, savepath=None):
    # colnames = ['acc_kmeans', 'recall_train', 'precision_train', 'f1_train', 'loss_value_train', 'accuracy_train',
    #             'recall_test', 'precision_test', 'f1_test', 'loss_value_test', 'accuracy_test', 'NMI_t_t1']

    f1_test = Result['f1_test'].values
    NMI_t_label = Result['NMI_t_label'].values
    NMI_t_t1 = Result['NMI_t_t1'].values
    # 画损失函数图
    fig = plt.figure(figsize=(10, 3))
    # 坐标轴的刻度设置向内(in)或向外(out)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    ax1 = fig.add_subplot(131)
    ax1.tick_params(top=True, right=True)
    ax1.plot(NMI_t_label)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('NMI t / labels')

    # subplot acc_train
    ax2 = fig.add_subplot(132)
    ax2.tick_params(top=True, right=True)

    ax2.plot(NMI_t_t1)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('NMI t-1 / t')

    ax3 = fig.add_subplot(133)
    ax3.tick_params(top=True, right=True)
    ax3.plot(f1_test)
    ax3.set_ylim([0.5, 1])
    # ax3.plot(accuracy_test, label='Test accuracy')

    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('$F_1$')

    plt.tight_layout()
    if savepath is None:
        plt.show()
    else:
        fig.savefig(savepath, dpi=600, format='png')
        plt.close(fig)


def decisionGraph(feature_all, show=False):
    if show:
        feature_all_n = (feature_all - feature_all.min(0)) / (feature_all.max(0) - feature_all.min(0) + 0.0001)
        dist_array = pdist(feature_all_n, metric='euclidean')
        dist_m = squareform(dist_array, 'tomatrix')
        NE = dist_m.shape[0]
        sort_distrow = np.sort(dist_array)
        dc = sort_distrow[int(NE * (NE - 1) / 2 * 0.02)]
        rho = np.sum(np.exp(-1 * (dist_m / dc) ** 2), axis=0) - 1

        delta = np.zeros([NE], np.float32)
        INN = np.inf * np.ones_like(delta, np.int32)
        rhoRho = np.argsort(-1 * rho)   # 降序的索引
        for rho_i in tqdm.tqdm(range(1, NE)):
            delta[rhoRho[rho_i]] = max(dist_m[rhoRho[rho_i], ...])
            for rho_j in range(rho_i):
                if delta[rhoRho[rho_i]] > dist_m[rho_i, rho_j]:
                    delta[rhoRho[rho_i]] = dist_m[rho_i, rho_j]
                    INN[rhoRho[rho_i]] = rhoRho[rho_j]
        delta[rhoRho[0]] = max(delta)
        INN[rhoRho[0]] = 0

        plt.figure()
        plt.scatter(rho, delta+0.01)
        # plt.xscale('log')
        # plt.yscale('log')
        plt.show()