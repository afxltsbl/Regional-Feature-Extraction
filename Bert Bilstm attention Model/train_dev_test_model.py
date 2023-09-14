import time
import torch
import d2l.torch as d2l
import numpy as np
from metrics import Metric
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.rcParams['font.sans-serif'] = ['SimHei']


def train_model(config, train_iter, valid_iter, net, loss, optimizer, num_epochs, writer):
    """Train and validation the best model."""
    return_result = []
    train_num = 0
    best_model_eval_micro_f1 = 0.0
    for epoch in range(num_epochs):
        counter = 0
        net.train()
        metric = d2l.Accumulator(4)  # 记录损失、整体准确率、标签准确率、数量
        for s, (inputs, mask, labels) in enumerate(train_iter):
            counter += 1
            train_num += 1
            if config.device == 'cuda':
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = net(inputs, mask)
            net.zero_grad()
            l = loss(outputs, labels)
            l.backward()
            optimizer.step()

            metrics = Metric(outputs.detach().numpy(), labels.detach().numpy())
            train_step_all_acc = metrics.accuracy_all()
            train_step_label_acc = metrics.accuracy_mean()
            metric.add(l, train_step_all_acc, train_step_label_acc, 1)
            print(f'【train】 epoch: {epoch + 1}/{num_epochs}, '
                  f'step: {counter}/{len(train_iter)}, '
                  f'train_loss:{round(l.item(), 5)}, '
                  f'train_all_acc:{round(train_step_all_acc, 5)}, '
                  f'train_label_acc:{round(train_step_label_acc, 5)}')
            writer.add_scalar('train_loss', l.item(), train_num)
            writer.add_scalar('train_all_acc', train_step_all_acc, train_num)
            writer.add_scalar('train_label_acc', train_step_label_acc, train_num)

            if (counter % config.print_stats == 0) or (s == len(train_iter) - 1):
                # 每过少轮输出在训练集和验证集上的效果，并评估是否保存模型
                eval_loss, eval_all_acc, eval_label_acc, eval_micro_f1, eval_macro_f1, eval_recall, classification_report = evaluate(valid_iter, net, config.loss, config)
                net.train()
                return_result.append(
                    [epoch, counter, l.item(), train_step_all_acc, eval_loss, eval_all_acc, eval_label_acc, eval_micro_f1, eval_macro_f1, eval_recall])
                print('【 dev 】 epoch: {}/{} '.format(epoch + 1, num_epochs),
                      'step: {}/{} '.format(counter, len(train_iter)),
                      'eval_loss: {:.5f} '.format(eval_loss),
                      'eval_all_acc: {:.5f} '.format(eval_all_acc),
                      'eval_label_acc: {:.5f} '.format(eval_label_acc),
                      'eval_micro_f1: {:.5f} '.format(eval_micro_f1),
                      'eval_macro_f1: {:.5f} '.format(eval_macro_f1),
                      'eval_recall: {:.5f} '.format(eval_recall),
                      time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
                print(classification_report)
                writer.add_scalar(f'eval_loss', eval_loss, train_num)
                writer.add_scalar(f'eval_all_acc', eval_all_acc, train_num)
                writer.add_scalar(f'eval_label_acc', eval_label_acc, train_num)
                writer.add_scalar(f'eval_micro_f1', eval_micro_f1, train_num)
                writer.add_scalar(f'eval_macro_f1', eval_macro_f1, train_num)
                writer.add_scalar(f'eval_recall', eval_recall, train_num)

                if eval_micro_f1 > best_model_eval_micro_f1:
                    # evaluate macro_f1 and save the best model
                    checkpoint = {
                        'epoch': epoch,
                        'loss': eval_loss,
                        'eval_all_acc': eval_all_acc,
                        'eval_label_acc': eval_label_acc,
                        'eval_micro_f1': eval_micro_f1,
                        'eval_macro_f1': eval_macro_f1,
                        'eval_recall': eval_recall,
                        'net_state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    best_model_eval_micro_f1 = eval_micro_f1
                    torch.save(checkpoint, config.model_save_path)
                    print('best model saved')

        # 每个epoch保存loss/精度并打印
        writer.add_scalar(f'Epoch_TRAIN_LOSS', metric[0] / metric[3], epoch)
        writer.add_scalar(f'Epoch_TRAIN_ALL_ACC', metric[1] / metric[3], epoch)
        writer.add_scalar(f'Epoch_TRAIN_LABELS_ACC', metric[2] / metric[3], epoch)
        print('【Epoch】 Epoch: {}/{} '.format(epoch + 1, num_epochs),
              'TRAIN_LOSS: {:.5f} '.format(metric[0] / metric[3]),
              'TRAIN_ALL_ACC: {:.5f} '.format(metric[1] / metric[3]),
              'TRAIN_LABELS_ACC: {:.5f} '.format(metric[2] / metric[3]),
              time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    writer.close()
    return return_result


def evaluate(valid_iter, net, loss, config):
    """evaluate model and return the loss and accuracy in valid dataset"""
    net.eval()
    loss_total = 0.0
    predict_all = []
    labels_all = []
    with torch.no_grad():
        for i, (x, mask, y) in enumerate(valid_iter):
            if config.device == 'cuda':
                x, y = x.cuda(), y.cuda()
            outputs = net(x, mask)
            l = loss(outputs, y)
            loss_total += l.item()
            predict_all.extend(outputs.detach().numpy().tolist())
            labels_all.extend(y.detach().numpy().tolist())
    metric = Metric(np.array(predict_all), np.array(labels_all))
    eval_all_acc = metric.accuracy_all()
    eval_label_acc = metric.accuracy_mean()
    # return valid dataset loss, all_acc, label_acc, micro_f1, macro_f1, recall
    return loss_total / len(valid_iter), eval_all_acc, eval_label_acc, metric.micfscore(), metric.macfscore(), metric.recall(), metric.classification_report(config.label_list)


def test(net, checkpoint_path, test_iter, loss, config):
    """test model and return the loss and accuracy in test dataset"""
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['net_state_dict'])
    net.eval()
    total_loss = 0.0
    test_outputs = []
    test_targets = []
    with torch.no_grad():
        for i, (x, mask, y) in enumerate(test_iter):
            if config.device == 'cuda':
                x, y = x.cuda(), y.cuda()
            output = net(x, mask)
            l = loss(output, y)
            total_loss += l.item()
            test_outputs.extend(output.detach().numpy().tolist())
            test_targets.extend(y.detach().numpy().tolist())
    metric = Metric(np.array(test_outputs), np.array(test_targets))
    # plot multilabel confusion matrix
    matrix = metric.multilabel_confusion_matrix()
    fig, ax = plt.subplots(1, 5, figsize=(22, 5))
    for axes, cfs_matrix, label in zip(ax.flatten(), matrix, config.label_list):
        print_confusion_matrix(cfs_matrix, axes, label, ["N", "Y"])
    fig.tight_layout()
    plt.savefig(config.multilabel_matrix_png)
    plt.show()
    # return test dataset loss, all_acc, label_acc, micro_f1, macro_f1, recall, classification_report
    return test_outputs, total_loss, metric.accuracy_all(), metric.accuracy_mean(), metric.micfscore(), metric.macfscore(), metric.recall(), metric.classification_report(config.label_list), metric.multilabel_confusion_matrix()

def print_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=8):

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')
    axes.set_title("Confusion Matrix for the class - " + class_label)


def predict(net, checkpoint_path, pre_iter, config):
    """predict data and return the predict result"""
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['net_state_dict'])
    net.eval()
    pre_outputs = []
    with torch.no_grad():
        for i, (x, mask) in enumerate(pre_iter):
            output = net(x, mask)
            pre_outputs.extend(output.detach().numpy().tolist())
            if i % 32 == 0:
                print(f'predict {i} columns...')
    pre_outputs = np.where(np.array(pre_outputs) > 0.5, 1, 0)
    return pre_outputs
