# !pip install setfit
import torch
import random
import numpy as np
from sentence_transformers.losses import CosineSimilarityLoss
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score,classification_report,multilabel_confusion_matrix
from setfit import SetFitTrainer,SetFitModel
import pyarrow as pa
import pandas as pd
from datasets import Dataset
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, StratifiedKFold
from statistics import mean 

from sklearn import metrics

seed = 2024
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():  
  device = "cuda:0" 
else:  
  device = "cpu"  

## Dataset
df = pd.read_excel("PE_combined.xlsx")
# df.drop(columns=['Unnamed: 0','Accession Number'],inplace=True)
df['doc'] = df['doc'].replace(r'\s+|\\n', ' ', regex=True) 
df['target'] = df['target'].replace({'Definitive PE NEG':'0'})
df['target'] = df['target'].replace({'Definitive PE POS':'2'})
df['target'] = df['target'].replace({'Probable PE NEG':'1'})
df['target'] = df['target'].replace({'Probable PE POS':'1'})
df['target'] = df['target'].replace({'Indeterminate':'inconclusive'})
df['target'] = df['target'].replace({'Non-diagnostic':'inconclusive'})
df = df[df['target'] != 'inconclusive']
train_df,test_df = train_test_split(df, test_size=0.2,stratify=df['target'],random_state=seed)
# train_df , eval_df = train_test_split(train_df, test_size=0.25,stratify=train_df['target'],random_state=20)


def cal_metric_boost(predictions_,truth_,probility,times):
    rng = np.random.RandomState(seed=202)
    idx = np.arange(truth_.shape[0])

    test_accuracies = []
    sensitivities = []
    specificities = []
    f1s = []
    aucs =[]
    recalls = []
    for i in range(times):

        pred_idx = rng.choice(idx, size=idx.shape[0], replace=True)
        acc_test_boot = np.mean(predictions_[pred_idx] == truth_[pred_idx])
        mcm = multilabel_confusion_matrix(predictions_[pred_idx], truth_[pred_idx])

        # tps, tns, fps, and fns are 1D arrays of shape (n_classes)
        tps = mcm[:, 1, 1]
        tns = mcm[:, 0, 0]
        fps = mcm[:, 0, 1]
        fns = mcm[:, 1, 0]

        # specificity and sensitivity are 1D arrays of shape (n_classes)
        specificity = tns / (tns + fps)
        # sensitivity = tps / (tps + fns)
        f1 = f1_score(truth_[pred_idx],predictions_[pred_idx],average='macro')
        recall = recall_score(truth_[pred_idx],predictions_[pred_idx],average='macro')
        try:
            auc = roc_auc_score(truth_[pred_idx],probility[pred_idx],multi_class="ovr",average="macro")
        except:
            print('_____________________')
        test_accuracies.append(acc_test_boot)
        # sensitivities.append(np.mean(sensitivity))
        # print('Specificity:', recall_score(np.logical_not(y_test),np.logical_not(y_pred),average='macro'))
        specificities.append(np.mean(specificity))
        f1s.append(f1)
        recalls.append(recall)
        aucs.append(auc)



    ci_lower_acc = np.percentile(test_accuracies, 2.5)
    ci_upper_acc = np.percentile(test_accuracies, 97.5)

    # ci_lower_ses = np.percentile(sensitivities, 2.5)
    # ci_upper_ses = np.percentile(sensitivities, 97.5)

    ci_lower_spe = np.percentile(specificities, 2.5)
    ci_upper_spe = np.percentile(specificities, 97.5)

    ci_lower_f1 = np.percentile(f1s, 2.5)
    ci_upper_f1 = np.percentile(f1s, 97.5)

    ci_lower_recall = np.percentile(recalls, 2.5)
    ci_upper_recall = np.percentile(recalls, 97.5)

    ci_lower_auc = np.percentile(aucs, 2.5)
    ci_upper_auc = np.percentile(aucs, 97.5)

    print('acc 95%CI:', ci_lower_acc, ci_upper_acc,'\n',\
        # 'sensi 95%CI:', ci_lower_ses, ci_upper_ses,'\n',\
        'spe 95%CI:', ci_lower_spe, ci_upper_spe,'\n',\
        'f1 95%CI:', ci_lower_f1, ci_upper_f1,'\n',\
        'recall 95%CI:', ci_lower_recall, ci_upper_recall,'\n'\
        'auc 95%CI:', ci_lower_auc, ci_upper_auc,'\n')

## Model
def model_init_roberta(params):
    params = params or {}
    return SetFitModel.from_pretrained("all-roberta-large-v1", **params)


    

def compute_metrics(labels,pred):
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred,average='macro')
    precision = precision_score(y_true=labels, y_pred=pred,average='macro')
    f1 = f1_score(y_true=labels, y_pred=pred,average='macro')
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

## Hyperparameter Search
skf = StratifiedKFold(n_splits=5)
iterations_test = [10,15,20,25]
batch_test = [4,8,16]

hyperparameter_search_results = []
for num_iterations in iterations_test:
    for batch_size in batch_test:
        list_of_results = []
        for train_index, test_index in skf.split(train_df['doc'], train_df['target']):

            train_dataset = Dataset(pa.Table.from_pandas(train_df.iloc[train_index]))
            eval_dataset = train_df.iloc[test_index]
            # eval_dataset = Dataset(pa.Table.from_pandas(train_df.iloc[test_index]))
            trainer = SetFitTrainer(
                model_init=model_init,
                train_dataset=train_dataset,
                # eval_dataset=eval_dataset,
                loss_class=CosineSimilarityLoss,
                batch_size=batch_size,
                num_iterations=num_iterations, # The number of text pairs to generate for contrastive learning
                num_epochs=1, # The number of epochs to use for constrastive learning
                column_mapping={"doc": "text", "target": "label"},
                seed=123,
            )
            trainer.train()

            preds = trainer.model.predict(list(eval_dataset['doc']))
            truth = list(eval_dataset['target'])
            results = compute_metrics(truth,preds)
            list_of_results.append(results)
        avg_results = {}
        for key in results.keys():
            avg_results[key] = mean([d[key] for d in list_of_results])
        hyperparameter_search_results.append(avg_results)


# Eval 


from sklearn.metrics import classification_report

truth_l=[]
pred_l=[]

for seed in [426]:
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_dataset = Dataset(pa.Table.from_pandas(train_df))

    trainer = SetFitTrainer(
        model_init=model_init_roberta,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        loss_class=CosineSimilarityLoss,
        batch_size=16,
         # The number of iterations to generate text pairs for contrastive learning
        num_epochs=1, # The number of epochs to use for constrastive learning
        column_mapping={"doc": "text", "target": "label"},
        seed = seed
    )
    trainer.args.sampling_strategy='oversampling',
    trainer.train()
    preds = trainer.model.predict(list(test_df['doc']))
    prob = trainer.model.predict_proba(list(test_df['doc']))
    truth = np.array(test_df['target'])
    results = compute_metrics(truth,preds)
    macro_roc_auc_ovr = roc_auc_score(truth,prob,multi_class="ovr",average="macro")
    print(f"Macro-averaged One-vs-Rest ROC AUC score:\n{macro_roc_auc_ovr:.2f}")
    print(results)
    print(classification_report(truth,preds))
    cal_metric_boost(preds,truth,prob, 2000)