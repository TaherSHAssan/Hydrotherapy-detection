import os
from os import walk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import seaborn as sns
import random
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.functional import accuracy
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
logger_path = cwd + '\\Trained Model Data\\logs'
model_path = cwd + '\\Trained Model Data\\classification.pth'
train_files_path = cwd + '\\TRAIN\\'
test_files_path = cwd + '\\TEST\\'
results_path = cwd + '\\Trained Model Data\\Results\\'
train_filenames = next(walk(train_files_path), (None, None, []))[2]
test_filenames = next(walk(test_files_path), (None, None, []))[2]
directory = train_files_path
# Set your target row number
target_row_count = 160  # Change this to your desired number of rows
# Update each file to have the desired number of rows
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r', newline='') as file:
            reader = csv.reader(file)
            rows = [row for row in reader if row]  # Exclude empty rows
        # Truncate or keep the file as is, based on the target row count
        rows = rows[:target_row_count]
        # Write the updated data back to the file
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)
def data_list(files_path, filenames):
    Abduction_data_list = []
    FE_data_list = []
    HR_data_list = []
    HSH_data_list = []
    Jogging_data_list = []
    RC_data_list = []
    SLR_data_list = []
    Squats_data_list = []
    Walking_data_list = []
    num = int(len(filenames) / 9)
        predict_abduction = pd.DataFrame(predictions_abduction)
        predict_FE = pd.DataFrame(predictions_FE)
        predict_HR = pd.DataFrame(predictions_HR)
        predict_HSH = pd.DataFrame(predictions_HSH)
        predict_Jogging = pd.DataFrame(predictions_jogging)
        predict_RC = pd.DataFrame(predictions_RC)
        predict_SLR = pd.DataFrame(predictions_SLR)
        predict_Squats = pd.DataFrame(predictions_Squats)
        predict_Walking = pd.DataFrame(predictions_Walking_data)
        Sequences = []
        for series_id, group in x.groupby('SeriesID'):
            sequence_features = group[FeatureColumns]
            label = y[y.SeriesID == series_id].iloc[0].Label
            Sequences.append((sequence_features, label))
        for seq in Sequences:
            sequences.append(seq)
    return sequences, FeatureColumns, label_encoder
N_EPOCHS = 50
BATCH_SIZE = 128
data_module = GaitDataModule(train_sequences, test_sequences, BATCH_SIZE)
class SequenceModel(nn.Module):
    def __init__(self, n_features, n_classes, n_hidden=256, n_layers=4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=0
        )
        self.classifier = nn.Linear(n_hidden, n_classes)
class GaitPredictor(pl.LightningModule):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.model = SequenceModel(n_features, n_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters()
    def forward(self, x, labels=None):
        output = self.model(x)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output
    def training_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label']
        loss, outputs = self(sequences, labels)
        predictions = torch.argmax(outputs, dim=1)
        step_accuracy = accuracy(predictions, labels)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        self.log('train_accuracy', step_accuracy, prog_bar=True, logger=True)
        return {'loss': loss, 'accuracy': step_accuracy}
    def validation_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label']
        loss, outputs = self(sequences, labels)
        predictions = torch.argmax(outputs, dim=1)
        step_accuracy = accuracy(predictions, labels)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_accuracy', step_accuracy, prog_bar=True, logger=True)
        return {'loss': loss, 'accuracy': step_accuracy}
    def test_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label']
        loss, outputs = self(sequences, labels)
        predictions = torch.argmax(outputs, dim=1)
        step_accuracy = accuracy(predictions, labels)
        self.log('test_loss', loss, prog_bar=True, logger=True)
        self.log('test_accuracy', step_accuracy, prog_bar=True, logger=True)
        return {'loss': loss, 'accuracy': step_accuracy}
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.0001)
model = GaitPredictor(
    n_features=len(FeatureColumns),
    n_classes=len(label_encoder.classes_)
)
trainer = pl.Trainer(
    logger=logger,
    callbacks=[checkpoint_callback],
    max_epochs=N_EPOCHS,
    gpus=1,
    progress_bar_refresh_rate=30,
    deterministic=True
)
trainer.fit(model, data_module)
trainer.test(model, data_module)
checkpoints_path = cwd + '\\checkpoints\\'
trained_model = GaitPredictor.load_from_checkpoint(
    checkpoint_path=checkpoints_path + '\\best_checkpoints.ckpt\\',
    n_features=len(FeatureColumns),
    n_classes=len(label_encoder.classes_)
)
trained_model.freeze()
test_dataset = HydrotherapyDataset(test_sequences)
predictions = []
labels = []
for item in tqdm(test_dataset):
    sequence = item['sequence']
    label = item['label']
    _, output = trained_model(sequence.unsqueeze(dim=0))
    prediction = torch.argmax(output, dim=1)
    predictions.append(prediction.item())
    labels.append(label.item())
report = classification_report(labels, predictions, target_names=label_encoder.classes_, output_dict=True)
Report = pd.DataFrame(report).transpose()
Report.to_csv(results_path + 'Classification_Report_Hyp.csv')
results = {'Labels': labels, 'Predictions': predictions}
Results = pd.DataFrame(results)
Results.to_csv(results_path + 'Results_Hyp_4.csv', index=False)
def show_confusion_matrix(confusion_matrix):
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16})
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=8)
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=8)
    plt.ylabel('True Exercise Type', fontsize=8)
    plt.xlabel('Predicted Exercise Type', fontsize=8)
cm = confusion_matrix(labels, predictions)
df_cm = pd.DataFrame(
    cm, index=label_encoder.classes_,
    columns=label_encoder.classes_)
show_confusion_matrix(df_cm)
torch.save(trained_model.state_dict(), model_path)
print('test_loss', 'loss', 'accuracy', 'test_accuracy')