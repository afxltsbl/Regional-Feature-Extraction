import time
import torch
import os
import pandas as pd
from transformers import BertTokenizer
from tensorboardX import SummaryWriter

import config
import pretreat
import train_dev_test_model
from models import Bert as model


if __name__ == '__main__':
    print('start run', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # 1.加载参数
    config = config.Config()

    # 2.加载数据
    data = pd.read_csv(os.path.join(config.data_path), encoding='utf-8', low_memory=False)
    data1, data2 = pretreat.split_data(data, config.col_select1, config.select1_num, config.col_select2, config.select2_num)    # data1：train, eval, test  data2:predict
    print('add data success, labels info:', '\n', data1[config.label_list].sum().sort_values())
    textlen = data1[config.content_name].apply(lambda x: len(x))
    print('text len info:', textlen.describe())

    # 3.预处理数据
    text_content = pretreat.pretreat_data(data1, config.content_name)
    mutil_labels = pretreat.multi_label_data(data1, config.label_list)  # 多标签

    # 4.加载预训练BERT的Tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.bert_dir)

    # 5.将文本转换为token
    result_comments_id = tokenizer(text_content, padding=True, truncation=True, max_length=config.max_length, return_tensors='pt')
    X_ids = result_comments_id['input_ids']
    X_mask = result_comments_id['attention_mask']

    # 6.划分训练数据和测试数据 备注：要打乱顺序，否则每次训练标签都一样
    X_ids_train, X_mask_train, Y_train, X_ids_test, X_mask_test, Y_test = pretreat.bert_train_test_split(X_ids, X_mask, mutil_labels, test_size=0.3, shuffle=True, random_state=1)
    X_ids_valid, X_mask_valid, Y_valid, X_ids_test, X_mask_test, Y_test = pretreat.bert_train_test_split(X_ids, X_mask, mutil_labels, test_size=0.2, shuffle=True, random_state=1)

    # 7.构建迭代器
    train_data = torch.utils.data.TensorDataset(X_ids_train, X_mask_train, Y_train)  # (([[],])，([[],]))
    train_iter = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True, drop_last=True)
    valid_data = torch.utils.data.TensorDataset(X_ids_valid, X_mask_valid, Y_valid)
    valid_iter = torch.utils.data.DataLoader(valid_data, batch_size=config.batch_size, shuffle=True, drop_last=True)
    test_data = torch.utils.data.TensorDataset(X_ids_test, X_mask_test, Y_test)
    test_iter = torch.utils.data.DataLoader(test_data, batch_size=config.batch_size, shuffle=True, drop_last=True)

    # 8.构建模型
    print('start training → loading model...', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    net = model.get_model(config)

    # 9.训练模型
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    writer = SummaryWriter(log_dir=config.log_dir)
    train_metrics = train_dev_test_model.train_model(config, train_iter, valid_iter,  net, config.loss, optimizer, config.num_epochs, writer)
    name = ['epoch', 'counter', 'train_loss', 'train_acc', 'eval_loss', 'eval_all_acc', 'eval_label_acc', 'eval_micro_f1', 'eval_macro_f1', 'eval_recall']
    train_metrics = pd.DataFrame(columns=name, data=train_metrics)
    train_metrics.to_csv(config.train_metrics_path, index=False, encoding='utf-8_sig')

    # 10.测试模型
    # print('start testing...', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # net = model.get_model(config)
    # checkpoint_path = config.checkpoint_path
    # test_outputs, test_loss, test_all_acc, test_label_acc, test_micro_f1, test_macro_f1, test_recall, classification_report, multilabel_confusion_matrix = train_dev_test_model.test(net, checkpoint_path, test_iter, config.loss, config)
    # y_pred = np.where(np.array(test_outputs) > 0.5, 1, 0)
    # data3 = pd.DataFrame(y_pred)
    # data1.to_excel('data1.xlsx', encoding='utf-8_sig', index=False)
    # data3.to_excel('data3.xlsx', encoding='utf-8_sig', index=False)
    # print('【test】 Test Loss: {:.5f} '.format(test_loss),
    #       'Test All Acc: {:.5f} '.format(test_all_acc),
    #       'Test label Acc: {:.5f} '.format(test_label_acc),
    #       'Test Micro F1: {:.5f} '.format(test_micro_f1),
    #       'Test Macro F1: {:.5f} '.format(test_macro_f1),
    #       'Test Recall: {:.5f} '.format(test_recall),
    #       time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # print(classification_report)

    # 11.预测
    # print('start predicting...', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # # print(data2['博文'])
    # predict_data = pretreat.pretreat_data(data2, '博文')
    # result_comments_id = tokenizer(predict_data, padding=True, truncation=True, max_length=config.max_length, return_tensors='pt')
    # pre_X = result_comments_id['input_ids']
    # pre_mask = result_comments_id['attention_mask']
    # X_ids_pre, X_mask_pre, Y_temp, X_ids_temp, X_mask_temp, Y_temp = pretreat.bert_train_test_split(pre_X, pre_mask,
    #                                                                                                      pre_mask,
    #                                                                                                      test_size=0,
    #                                                                                                      shuffle=True,
    #                                                                                                      random_state=3)
    # pre_data = torch.utils.data.TensorDataset(X_ids_pre, X_mask_pre)
    # pre_iter = torch.utils.data.DataLoader(pre_data, batch_size=config.batch_size, shuffle=False, drop_last=True)
    # net = model.get_model(config)
    # checkpoint_path = config.checkpoint_path
    # predict_outputs = train_dev_test_model.predict(net, checkpoint_path, pre_iter, config)
    #
    # temp = pd.DataFrame(predict_outputs)
    # temp.to_csv(r'temp1.csv', index=False, encoding='utf-8_sig')
    # data2.to_excel(r'temp2.xlsx', index=False, encoding='utf-8_sig')
    #
    # data2.iloc[:data2.shape[0]-2, 4:9] = temp
    # data2.to_excel('temp3.xlsx', index=False, encoding='utf-8_sig')  # 考虑是否保留序号，好和以前的数据对上
    # result = pd.concat([data1, data2])
    # result.to_excel('temp4.xlsx', index=False, encoding='utf-8_sig')