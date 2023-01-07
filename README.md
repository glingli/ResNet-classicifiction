# ResNet-classicifiction
一个用ResNet18来分类猫狗图片的项目. 只需要PyTorch框架即可（CPU版）。i5-12500h训练时间约4min，是一个相当轻量的网络。识别成功率高达98%。喜欢请右上角Star🌟一下，谢谢支持❤。
## 背景
由于网络训练需要数据集的支持，为了获取合适且足够的数据集，我们将专项任务的方向指向于各大竞赛包括数模、阿里天池等。同时通过ResNet相关论文的查阅和比赛题目及数据集的收集，我们最终将目光放在了Kaggle竞赛中使用ResNet对猫狗进行二分类的题目，并下载数据集。
## 总体思想
模型初始化方面，我们使用ResNet18网络结构，在PyTorch框架下进行模型的训练。定义ResNet模型，并且定义训练函数以及测试函数。在进行网络参数的初始化时，设置batch大小为32，图像输入大小使用默认的224，学习率使用固定值0.001。优化器选择随机梯度下降法，定义损失函数为交叉熵损失函数。这个函数就是专门求出分类问题的loss的，我们将输入图片经过神经网络处理后得到属于每种类的可能性和这个图片真实属于的类，并交叉熵损失函数会计算出loss。
	训练集和测试集处理方面，首先进行进行数据集的预处理，从下载数据集中猫、狗图片各1000张作为训练集，各500张作为测试集。处理时遵循统一大小、转化张量和正则化的处理流程。统一大小是由于图像输入大小设置为224，而数据集内的图像大多不符合要求，所以对输入的图片采取随机裁剪后放大或缩小到指定大小的策略。在尺寸处理完成后，将图片转化为可以由网络使用的张量形式。通过正则化逐通道的对图像进行标准化（均值变为0，标准差变为1），可以加快模型的收敛，标准化参数由训练集中抽样算出。
	模型的训练和验证方面，epoch参数选为2，优化器及损失函数由模型初始化中的参数传入，训练图片使用torch提供的DataLoader方法打包并传入。并且打印每个batch的loss和acc参数，同时打印时间参数以及其他辅助判断训练的消息。验证集采用另外猫狗图片各500张组成。
	测试部分，通过图片和尺寸大小输入模型，求输出向量，根据标签相似度，取相概率高的标签作为判断结果，从而对输入图片进行分类。在测试多张图片时，将路径下的所有图片遍历进行输入。
## 结构
![Untitled](https://user-images.githubusercontent.com/51522892/211127631-85451e8b-6fbc-4b5c-a052-3085e1be03c1.png)

Dog_Cat: 数据集

test: 实际希望分类的图片

train: 训练集

val: 测试集

work.py：参数控制及指令控制

newWork.py：训练结束后使用训练好的模型进行实际图片的分类

resnet.py：包含训练及测试使用的各个模块

数据集大约600Mb
下载链接：https://pan.baidu.com/s/1BwD3psguekhbJEErjFjdbA 
提取码：dycv 

喜欢请右上角Star一下，谢谢支持。
## 模型使用方法
读resnet.py里的work_model()方法，你就知道了
