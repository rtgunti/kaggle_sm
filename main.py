import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pretrainedmodels
import albumentations
from torch.nn import functional as F
from wtfml.data_loaders.image import ClassificationDataLoader
from wtfml.utils import EarlyStopping
from wtfml.engine import Engine
from sklearn import metrics

class SEResNext50_32x4d(nn.Module):
    def __init__(self, pretrained = "imagenet"):
        super(SEResNext50_32x4d, self).__init__()
        self.model = pretrainedmodels.se_resnext50_32x4d(pretrained=pretrained)
        self.out = nn.Linear(2048, 1)

    def forward(self, image, targets):
        bs, _, _, _ = image.shape
        x = self.model.features(image)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(bs, -1)
        out = self.out(x)
        loss = nn.BCEWithLogitsLoss()(out, targets.reshape(-1, 1).type_as(out))
        return out, loss

def train(fold):
    training_data_path = ""
    df = pd.read_csv("train_folds.csv")
    model_path = "/home/ravi/kaggle_sm/checkpoints"
    device = "cpu"
    epochs = 10
    train_bs = 32
    val_bs = 16
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_val = df[df.kfold == fold].reset_index(drop=True)

    train_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255, always_apply=True)
        ]
    )

    val_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255, always_apply=True)
        ]
    )

    train_images = df_train.image_name.values.tolist()
    train_images = [os.path.join(training_data_path, i + ".jpg") for i in train_images]
    train_targets = df_train.target.values


    val_images = df_val.image_name.values.tolist()
    val_images = [os.path.join(training_data_path, i + ".jpg") for i in val_images]
    val_targets = df_train.target.values

    train_dataset = ClassificationDataLoader(
        image_paths= train_images,
        targets = train_targets,
        resize=None,
        augmentations=train_aug
    )
    train_loder = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_bs,
        shuffle=True,
        num_workers=4
    )

    val_dataset = ClassificationDataLoader(
        image_paths= val_images,
        targets = val_targets,
        resize=None,
        augmentations=val_aug
    )

    val_loder = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_bs,
        num_workers=4
    )

    model = SEResNext50_32x4d(pretrained='imagenet')
    model.to(device)
    optimizer = torch.optims.Adam(model.parameters(), lr= 1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        mode="max"
    )
    engine = Engine(model, optimizer, device)
    es = EarlyStopping(patience=5, mode= "max")
    predictions = []
    for epoch in range(epochs):
        training_loss = engine.train(train_loder)
        val_loss = engine.evaluate(val_loder)
        predictions = np.vstack((engine.predict(val_loder))).ravel()
        auc = metrics.roc_auc_score(val_targets, predictions)
        scheduler.step(auc)
        print(f"epoch = {epoch}, auc = {auc}")
        es(auc, model, model_path)
        if es.early_stop:
            print("early stopping")
            break


if __name__ == "__main__":
    train(fold=0)