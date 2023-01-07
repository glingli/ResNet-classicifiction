
"""version 1.0"""
import copy
import os
import shutil
import time
from PIL import Image
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torchvision import transforms


def set_parameter_requires_grad(model, feature_extracting):
    """
    该函数用于将模型所有的梯度改为不可变
    :param model:要修改的模型
    :param feature_extracting:是否要改为不可变
    :return:
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    """
    初始化函数
    :param model_name: 模型名字
    :param num_classes: 输出神经元个数
    :param feature_extract = True
    :param use_pretrained = True
    :return:
    """
    if model_name == "resnet":
        model_ft = models.resnet18(pretrained=use_pretrained)

        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features

        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    return model_ft, input_size


def train_model(model, dataloaders, criterion, optimizer, num_epochs=5, log_interval=20):
    """
    训练模型
    :param model: 需要训练的模型
    :param dataloaders: 将图片打包处理并传入网络
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param num_epochs: number epoch
    :param log_interval: 每隔20batch输出，batch_size=32，则每隔640个数据输出
    :return model:完成训练的模型
    :return val_acc_history:精度
    """
    since = time.time()
    val_acc_history = []
    best_acc = 0.
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        running_loss = 0.
        running_corrects = 0.
        model.train()
        phase = 'train'
        for batch_id, (inputs, labels) in enumerate(dataloaders[phase]):
            with torch.autograd.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            optimizer.zero_grad()#梯度清0
            loss.backward()#反向传播，计算当前梯度；
            optimizer.step()#根据梯度更新网络参数
            if batch_id % log_interval == 0:
                print("Train Epoch: {} [{}/{} ({:0f}%)]\tLoss: {:.6f}\tAcc: {}/{}".format(
                    epoch,
                    batch_id * 32,
                    len(dataloaders['train'].dataset),
                    100. * batch_id / len(dataloaders['train']),
                    loss.item(),
                    int(running_corrects),
                    batch_id * 32
                ))
                epoch_loss, epoch_acc = test_model(model, dataloaders, criterion, epoch)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                val_acc_history.append(epoch_acc)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds.view(-1) == labels.view(-1)).item()

        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = running_corrects / len(dataloaders[phase].dataset)
        print("{} Loss: {} Acc: {}".format(phase, epoch_loss, epoch_acc))
        print()
    time_elapsed = time.time() - since #训练总耗时
    print("Training compete in {}m   {}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {}".format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def test_model(model, dataloaders, criterion, epoch):
    """
    测试模型
    :param model:
    :return 
    """
    running_loss = 0.
    running_corrects = 0.
    model.eval()#求解每种类的可能性
    since = time.time()
    for inputs, labels in dataloaders['val']:
        with torch.autograd.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds.view(-1) == labels.view(-1)).item()

    epoch_loss = running_loss / len(dataloaders['val'].dataset)
    epoch_acc = running_corrects / len(dataloaders['val'].dataset)
    time_elapsed = time.time() - since
    print("Training compete in {}m   {}s".format(time_elapsed // 60, time_elapsed % 60))
    print("{} Loss: {} Acc: {}".format('val', epoch_loss, epoch_acc))
    # model.load_state_dict(best_model_wts)
    return epoch_loss, epoch_acc

def work_model(model, data_dir, input_size):
    """
    使用已经训练好的模型对图片进行分类
    :param model: 已经训练好的模型
    :param data_dir: 项目路径
    :param input_size: 图片输入的大小
    :return:
    """
    result_path = data_dir + '/result/'
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.mkdir(result_path)
    os.mkdir(result_path + '/Dog')
    os.mkdir(result_path + '/Cat')
    test_path = data_dir + '/test/'
    for name in os.listdir(test_path):
        show('----name:',name,'----')
        image_name = test_path + name
        image = loader(image_name, input_size)
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        if preds == 1:
            shutil.copyfile(image_name, result_path + 'Dog/' + name)
        else:
            shutil.copyfile(image_name, result_path + 'Cat/' + name)


def loader(image_name, input_size):
    """
    function：将图片转化为可测试的tensor类型
    :param imagepath: 要测试的图片路径
    :return: 图片的tensor值
    """
    loader = torchvision.transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # loader使用torchvision中自带的transforms函数

    image = Image.open(image_name).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(torch.float)
    
def showResult(data_dir):
"""
    function：将分类结果进行展示
    :param data_dir: 要测试的图片路径
    :return: 分类正确数与总数
    """
    catWrong = 0
    dogWrong = 0
    counterC = 0
    counterD = 0
    result_path = data_dir + '/result/'
    cat_path = result_path + '/Cat'
    dog_path = result_path + '/Dog'
    for name in os.listdir(cat_path):
        counterC = counterC + 1
        
        if(name[0:6]=='[Cat_]'):
            catWrong = catWrong + 1
            
    for name in os.listdir(dog_path):
        counterD = counterD + 1
        if(name[0:6]=='[Dog_]'):
            dogWrong = dogWrong + 1
            
    return catWrong, dogWrong, counterC, counterD
        

if __name__ == '__main__':
    print(loader("./Dog_Cat/test/"))