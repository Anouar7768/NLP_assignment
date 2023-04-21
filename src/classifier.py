from typing import List

import pandas as pd
from tqdm import tqdm
import numpy as np

from helper_functions import add_labels, add_processed_col
from TransformerBinaryClassifier import TransformerBinaryClassifier

import torch
from datasets import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import Adam

from transformers import get_scheduler
from transformers import DataCollatorWithPadding


class Classifier(torch.nn.Module):
    """
    The Classifier: complete the definition of this class template by providing a constructor (i.e. the
    __init__() function) and the 2 methods train() and predict() below. Please donot change
     """

    def __init__(self, plm_name='roberta-base', num_epochs=20, lr=1e-6, batch_size=32, model=None):
        super(Classifier, self).__init__()
        self.plm_name = plm_name
        self.num_epochs = num_epochs
        self.lr = lr
        self.model = model
        self.batch_size = batch_size

    def tokenize_function(self, examples):
        return self.model.lmtokenizer(examples["processed_input"], truncation=True, add_special_tokens=True)

    ############################################# comp
    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the classifier model on the training set stored in file trainfile
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
          - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """
        ### Read training dataset and preprocess it
        df_train = pd.read_csv(train_filename, sep='\t', header=None)
        df_train = add_labels(df_train)
        df_train = add_processed_col(df_train)

        self.model = TransformerBinaryClassifier(self.plm_name)

        ds_train = Dataset.from_pandas(df_train[["labels", "processed_input"]])

        ### tokenize datasets
        tok_ds_train = ds_train.map(self.tokenize_function, batched=True)

        tok_ds_train = tok_ds_train.remove_columns(["processed_input"])

        data_collator = DataCollatorWithPadding(tokenizer=self.model.lmtokenizer,
                                                padding=True,
                                                return_tensors='pt')

        class_sample_count = np.array(
        [len(np.where(df_train["labels"] == t)[0]) for t in np.unique(df_train["labels"])])

        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in df_train["labels"]])
        samples_weight = torch.from_numpy(samples_weight)  

        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

        train_dataloader = DataLoader(tok_ds_train,
                                      shuffle=False,
                                      batch_size=self.batch_size,
                                      collate_fn=data_collator,
                                      sampler=sampler)

        optimizer = Adam(self.model.parameters(), lr=self.lr)

        num_training_steps = self.num_epochs * len(train_dataloader)

        lr_scheduler = get_scheduler(name="linear",
                                     optimizer=optimizer,
                                     num_warmup_steps=0,
                                     num_training_steps=num_training_steps)

        self.model.to(device)

        self.model.train()

        progress_bar = tqdm(range(num_training_steps))

        for epoch in range(self.num_epochs):
            for batch in train_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                predictions = self.model(batch)
                # labels = labels.type(torch.LongTensor)
                loss = self.model.loss_fn(predictions, batch['labels'])
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                print(loss)

        return None

    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """

        df = pd.read_csv(data_filename, sep='\t', header=None)
        df = add_labels(df)
        df = add_processed_col(df)

        self.model.eval()

        labels = ["negative", "neutral", "positive"]

        encoded_texts = self.model.lmtokenizer(list(df["processed_input"]),
                                               truncation=True,
                                               padding=True,
                                               return_attention_mask=True,
                                               return_tensors='pt',
                                               add_special_tokens=True)
        with torch.no_grad():
            output = self.model(encoded_texts.to(device)).tolist()
            pred_labels = [labels[np.argmax(p)] for p in output]
            return pred_labels
