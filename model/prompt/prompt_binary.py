#! -*- coding:utf-8 -*-

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import AutoModelForMaskedLM, AutoModel, AutoTokenizer, BertTokenizer, BertForMaskedLM, AdamW, Trainer, TrainingArguments, RobertaTokenizer, RobertaForMaskedLM
from datasets import Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import logging
import argparse
import torch
import torch.nn as nn


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
prefix_pattern = {'1':['This message is verifiable factual? ',', '], '2':['This message contains false information? ',', '],
                '3':['Will the claim be of interest to the public? ',', '], '4':['This message is harmful to others? ', ', '],
                '5':['Should this message be verified? ', ', '], '6':['Is this message harmful to society? ',', '],
                '7':['This message should get attention of government? ', ' .']}
answer = {'1':['no', 'yes'],'2':['no', 'yes'],'3':['no', 'yes'],'4':['no', 'yes'],'5':['no', 'yes'],'6':['no', 'yes'],'7':['no', 'yes']}
position = {}
appendix_pattern = {1:['']}
logging.basicConfig(level=logging.INFO)

class LecCallTag():
    # data process
    def data_process(self, data_file, qid, eda):
        if eda == 't':
            column_index = 'label'
        else:
            column_index = 'q'+qid+'_label'
        df = pd.read_csv(data_file, usecols=['text',column_index])
        df = df.drop(df[df[column_index]==-1].index)

        global class_weights
        class_weights = (1-(df[column_index].value_counts().sort_index() / len(df))).values
        class_weights = torch.from_numpy(class_weights).float().to("cuda")

        train_data = np.array(df)
        train_data_list = train_data.tolist()
        for pair in train_data_list:
            pair[1] = prefix_pattern[qid][0] + answer[qid][pair[1]] + prefix_pattern[qid][1] + pair[0]
            pair[0] = prefix_pattern[qid][0] + '[MASK]' + prefix_pattern[qid][1] + pair[0]
        df = pd.DataFrame(train_data_list)

        text = df[0].tolist()
        label = df[1].tolist()
        return text, label
    
    # model, tokenizer
    def create_model_tokenizer(self, model_name):
        model_name = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForMaskedLM.from_pretrained(model_name)
#        new_tokens = ['covid', 'covid19', '[url]', '[username]', 'pandemic', 'c19', 'covid-19']
#        num_added_toks = tokenizer.add_tokens(new_tokens)
 #       model.resize_token_embeddings(len(tokenizer))

        return tokenizer, model
    
    # consturct dataset
    def create_dataset(self, text, label, tokenizer, max_len):
        X_train, X_temp, Y_train, Y_temp = train_test_split(text, label, test_size=0.3, random_state=1)
        X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.6667, random_state=1)
        logging.info('train set: %s items, \nval set: %s itmes, \nteset set: %s itmes' %(len(X_train), len(X_val), len(X_test)))
        train_dict = {'text': X_train, 'label_text': Y_train}
        val_dict = {'text': X_val, 'label_text': Y_val}
        test_dict = {'text': X_test, 'label_text': Y_test}
        train_dataset = Dataset.from_dict(train_dict)
        val_dataset = Dataset.from_dict(val_dict)
        test_dataset = Dataset.from_dict(test_dict)
        def preprocess_function(examples):
            text_token = tokenizer(examples['text'], padding='max_length',truncation=True, max_length=max_len)
            text_token['labels'] = np.array(tokenizer(examples['label_text'], padding='max_length',truncation=True, max_length=max_len)["input_ids"])  # 注意数据类型
            return text_token
        train_dataset = train_dataset.map(preprocess_function, batched=True)
        val_dataset = val_dataset.map(preprocess_function, batched=True)
        test_dataset = test_dataset.map(preprocess_function, batched=True)
        return train_dataset, val_dataset, test_dataset
    
    # construct trainer
    def create_trainer(self, model, train_dataset, val_dataset, checkpoint_dir, epoch, batch_size, tokenizer):
        class WeightedLossTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                outputs = model(**inputs)

                logits = outputs.get("logits")[:, position[0]][:, [2053,2748]]
                labels = inputs.get("labels")[:, position[0]]
                labels = torch.where(labels==2053, 0, 1)
                loss_func = nn.CrossEntropyLoss(weight=class_weights)
                loss = loss_func(logits, labels)
                return (loss, outputs) if return_outputs else loss
        args = TrainingArguments(
            checkpoint_dir,
            evaluation_strategy = "epoch",
            save_strategy = "epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epoch,
            weight_decay=0.01,
            metric_for_best_model='accuracy',
            load_best_model_at_end = True
        )
        def compute_metrics(pred):
            labels = pred.label_ids[:, position[0]]
            preds = pred.predictions[:, position[0]].argmax(-1)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
            acc = accuracy_score(labels, preds)
            report = classification_report(labels, preds, digits=4)
            logging.info(report)
            
            return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
        return trainer

def compute_postion(position, tokenizer, qid):
    position[0] = len(tokenizer(prefix_pattern[qid][0])['input_ids']) - 1

def test(test_dataset, trainer, qid):
    # Make prediction
    raw_pred = trainer.predict(test_dataset)
    labels = raw_pred.label_ids[:, position[0]]
    preds = raw_pred.predictions[:, position[0]].argmax(-1)

    cm = confusion_matrix(y_true=labels, y_pred=preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig('./results/'+'prompt_'+qid+'.png')#save image

    report = classification_report(labels, preds, digits=4)
    my_open = open('./results/'+'prompt_'+qid+'.txt', 'w')
    my_open.write(report)
    my_open.close()

def main(model_name, qid, epoch, eda):
    lct = LecCallTag()
    if eda == 't':
        data_file = f'./data/eda/{qid}_bin.csv'
    else :
        data_file = './data/aggre_binary.csv'
    checkpoint_dir = "./results/checkpoint/"
    batch_size = 32
    max_len = 120
    
    text, label = lct.data_process(data_file, qid, eda)
    tokenizer, model = lct.create_model_tokenizer(model_name)
    compute_postion(position, tokenizer, qid)
    train_dataset, val_dataset, test_dataset = lct.create_dataset(text, label, tokenizer, max_len)
    trainer = lct.create_trainer(model, train_dataset, val_dataset, checkpoint_dir, epoch, batch_size, tokenizer)
    trainer.train()

    test(test_dataset, trainer, qid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add models\' arguments')
    parser.add_argument('--type', default='bt',
                    help='use different pretrained model to train and test')
    parser.add_argument('--qid', default='1')
    parser.add_argument('--ep', default='10', help='number of epochs')
    parser.add_argument('--eda', default='f', help='data argumentation or not')
    args = parser.parse_args()

    main(args.type, args.qid, int(args.ep), args.eda)
