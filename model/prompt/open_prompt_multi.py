from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, MixedTemplate
from openprompt import PromptForClassification
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptDataLoader
import torch
import pandas as pd
from transformers import  AdamW, get_linear_schedule_with_warmup
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,BertTokenizer,Trainer, TrainingArguments
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_recall_fscore_support, accuracy_score
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser(description='Add models\' arguments')
parser.add_argument('--f', default='prefix', help='form type')
parser.add_argument('--qid', default='2', help='Question ID')
parser.add_argument('--eda', default='0', help='eda or not')
args = parser.parse_args()
FORM = args.f
QID = int(args.qid)

def load_data(filepath):
    df = pd.read_csv(filepath)
    ret = []
    for index, data in df.iterrows():
        label = int(data['q'+str(QID)+'_label'])
        if label != -1:
            input_example = InputExample(text_a = data['text'], label=label, guid=index)
            ret.append(input_example)
    return ret

dataset = {}
dataset['train'] = load_data('./data/prepro/train_multiclass.csv')
dataset['dev'] = load_data('./data/prepro/dev_multiclass.csv')
dataset['test'] = load_data('./data/prepro/test_multiclass.csv')

plm, tokenizer, model_config, WrapperClass = load_plm("bert", "princeton-nlp/sup-simcse-bert-base-uncased")
questions = ['','', 'To what extent does the message appear to contain false information?','Will the claim be of interest to the public?',
            'This message is harmful to others?', 'Should this message be verified?',
            'Is this message harmful to society?', 'This message should get attention of government?']
class_numbers = [-1, -1, 5, 5, 5, 4, 7, 10]
label_words = [
            [],[],
            [["no"],["little"],["maybe"],["much"],["yes"]],
            [["no"],["little"],["maybe"],["much"],["yes"]],
            [["no"],["little"],["maybe"],["much"],["yes"]],
            [["no"],["little"],["maybe"],["much"],["yes"]],
            [["no"],["joke"],["maybe"],["yes"],["much"],["bad"],["rumor","conspiracy"],["other"]],
            [["no"],["maybe"],["yes"],["other"],['blame'],['advice'],['action'],['take'],['cure'],['question']]
        ]
promptTemplate = {
    'suffix':ManualTemplate(
        text = '{"placeholder":"text_a"} '+questions[QID]+' {"mask"}.',
        tokenizer = tokenizer), 
    'prefix':ManualTemplate(
        text = questions[QID]+' {"mask"}.'+' {"placeholder":"text_a"}',
        tokenizer = tokenizer),
    'soft_prefix':MixedTemplate(
        text = '{"soft"} {"soft"} {"soft"} {"mask"} {"soft"} {"soft"} {"soft"}. {"placeholder":"text_a"}',
        model=plm, tokenizer = tokenizer),
    'soft_suffix':MixedTemplate(
        text = '{"placeholder":"text_a"} {"soft"} {"soft"}  {"mask"} {"soft"} {"soft"} .',
        model=plm, tokenizer = tokenizer)
}

# tokenizer = BertTokenizer(vocab_file='./data/vocab.txt')
wrapped_tokenizer = WrapperClass(max_seq_length=128, decoder_max_length=3, tokenizer=tokenizer,truncate_method="head")
model_inputs = {}
for split in ['train', 'dev', 'test']:
    model_inputs[split] = []
    for sample in dataset[split]:
        tokenized_example = wrapped_tokenizer.tokenize_one_example(promptTemplate[FORM].wrap_one_example(sample), teacher_forcing=False)
        model_inputs[split].append(tokenized_example)


promptVerbalizer = ManualVerbalizer(
    num_classes= class_numbers[QID],
    label_words = label_words[QID],
    tokenizer = tokenizer,
)

train_dataloader = PromptDataLoader(dataset=dataset["train"], template=promptTemplate[FORM], tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=128, decoder_max_length=3,
    batch_size=32,shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")
# Evaluate
validation_dataloader = PromptDataLoader(dataset=dataset["dev"], template=promptTemplate[FORM], tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=128, decoder_max_length=3,
    batch_size=32,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")

# start training
use_cuda = True
prompt_model = PromptForClassification(plm=plm,template=promptTemplate[FORM], verbalizer=promptVerbalizer, freeze_plm=False)
if use_cuda:
    prompt_model=  prompt_model.cuda()

# Now the training is standard
loss_func = torch.nn.CrossEntropyLoss()
no_decay = ['bias', 'LayerNorm.weight']
# it's always good practice to set no decay to biase and LayerNorm parameters
optimizer_grouped_parameters = [
    {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)
checkpoint_dir = "./results/checkpoint/best_model.pt"
acc = 0
for epoch in range(20):
    tot_loss = 0
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        tot_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        if step %100 ==1:
            print("Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)
    
    # Evaluation
    allpreds = []
    alllabels = []
    for step, inputs in enumerate(validation_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    cur_acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    if (cur_acc>acc):
        acc = cur_acc
        torch.save(prompt_model, checkpoint_dir)




# Test
test_dataloader = PromptDataLoader(dataset=dataset["test"], template=promptTemplate[FORM], tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=128, decoder_max_length=3,
    batch_size=32,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")

allpreds = []
alllabels = []
prompt_model = torch.load(checkpoint_dir)
for step, inputs in enumerate(test_dataloader):
    if use_cuda:
        inputs = inputs.cuda()
    logits = prompt_model(inputs)
    labels = inputs['label']
    alllabels.extend(labels.cpu().tolist())
    allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

cm = confusion_matrix(y_true=alllabels, y_pred=allpreds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig('./results/open_prompt_multi_'+FORM+'_'+str(QID)+'.png')#save image

report = classification_report(alllabels, allpreds, digits=4)
my_open = open('./results/open_prompt_multi_'+FORM+'_'+str(QID)+'.txt', 'w')
my_open.write(report)
my_open.close()