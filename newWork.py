import resnet
import torch

#initialize
pthfile = 'myResNet.pth'
model_data = torch.load(pthfile)
data_dir = "./Dog_Cat"
input_size = 224

#try 2 classicifaction only
resnet.work_model(model_data, data_dir, input_size)
catWrong, dogWrong, counterC, CounterD = resnet.showResult(data_dir)
accC = catWrong/counterC
print('catRight='+str(catWrong)+' in '+str(counterC))
print('----acc='+str(accC))
accD = dogWrong/counterD
print('DogRight='+str(dogWrong)+' in '+str(counterD))
print('----acc='+str(accD))
print('-------------------')
acc = (catWrong+dogWrong)/(counterC+counterD)
print('accuricy='+str(acc))
print('-------------------')
