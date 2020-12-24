import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import sys
from os import path
import numpy as np
from torch import optim
import os
import soundfile
from tqdm import tqdm_notebook as tqdm
import torch.nn as nn
import time
import pickle
import librosa
import numpy as np
from util import *
from model import *
from dataset import *

audio_data_path = "audio_data.pkl"
save_path = "spec_checkpoint.th"
epochs = 5
    
data = pickle.load(open(audio_data_path, "rb"))
train_dataset = ViolenceDataset(data)

model = ConvModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-05)
criterion = nn.CrossEntropyLoss().cuda()

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

best_acc = 0

for epoch in range(epochs):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    mi_meter = AverageMeter()
    temperature_meter = AverageMeter()
    end = time.time()
    model.train()
    for i, (inputs, targets) in enumerate(train_loader):
        """
        for idx in range(len(inputs)):
            inputs[idx] = spec_augment(inputs[idx])
        """
        inputs = inputs.cuda()
        targets = targets.cuda()
        output = model(inputs).squeeze(-1)

        loss = criterion(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        
        prec1 = accuracy(output.data, targets)[0]
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
    print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
    if top1.avg > best_acc:
        best_acc = top1.avg
        torch.save({
            'state_dict': model.state_dict(),
            }, save_path)