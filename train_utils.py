from collections import defaultdict

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np

from utils import AverageMeter
from metrics import categorical_accuracy


def _fit_epoch(model, data, criterion, optimizer, batch_size, shuffle):
    model.train()
    running_loss = AverageMeter()
    running_accuracy = AverageMeter()
    loader = DataLoader(data, batch_size, shuffle)
    t = tqdm(loader, total=len(loader))
    for data, target in t:
        data, target = Variable(data.cuda()), Variable(target.cuda())
        output = model(data)
        loss = criterion(output, target)
        accuracy = categorical_accuracy(target.data, output.data)
        running_loss.update(loss.data[0])
        running_accuracy.update(accuracy)
        t.set_description("[ loss: {:.4f} | acc: {:.4f} ] ".format(
            running_loss.avg, running_accuracy.avg))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return running_loss.avg, running_accuracy.avg


def fit(model, train, criterion, optimizer, batch_size=32,
        shuffle=True, nb_epoch=1, validation_data=None, cuda=True):
    # TODO: implement CUDA flags, optional metrics and lr scheduler
    if validation_data:
        print('Train on {} samples, Validate on {} samples'.format(len(train), len(validation_data)))
    else:
        print('Train on {} samples'.format(len(train)))

    history = defaultdict(list)
    t = tqdm(range(nb_epoch), total=nb_epoch)
    for epoch in t:
        loss, acc = _fit_epoch(model, train, criterion,
                              optimizer, batch_size, shuffle)

        history['loss'].append(loss)
        history['acc'].append(acc)
        if validation_data:
            val_loss, val_acc = validate(model, validation_data, criterion, batch_size)
            print("[Epoch {} - loss: {:.4f} - acc: {:.4f} - val_loss: {:.4f} - val_acc: {:.4f}]".format(epoch+1,
                                                                                                        loss,
                                                                                                        acc,
                                                                                                        val_loss,
                                                                                                        val_acc))
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
        else:
            print("[loss: {:.4f} - acc: {:.4f}]".format(loss, acc))
    return history


def validate(model, validation_data, criterion, batch_size):
    model.eval()
    val_loss = AverageMeter()
    val_accuracy = AverageMeter()
    loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
    for data, target in loader:
        data, target = Variable(data.cuda()), Variable(target.cuda())
        output = model(data)
        loss = criterion(output, target)
        accuracy = categorical_accuracy(target.data, output.data)
        val_loss.update(loss.data[0])
        val_accuracy.update(accuracy)
    return val_loss.avg, val_accuracy.avg


def predict(model, data, batch_size):
    model.eval()
    predictions = []
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    for data, _ in tqdm(loader, total=len(loader)):
        data = Variable(data.cuda())
        output = model(data).data
        for prediction in output:
            predictions.append(prediction.cpu().numpy())
    return np.array(predictions)


def predict_labels(model, data, batch_size):
    model.eval()
    labels = []
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    for data, _ in tqdm(loader, total=len(loader)):
        data = Variable(data.cuda())
        _, output = torch.max(model(data).data, dim=-1)
        for label in output:
            labels.append(label.cpu().numpy())
    return np.array(labels)


def generate_predictions(model, data, batch_size):
    model.eval()
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    for data, _ in loader:
        data = Variable(data.cuda())
        output = model(data).data
        for prediction in output:
            yield prediction.cpu().numpy()
