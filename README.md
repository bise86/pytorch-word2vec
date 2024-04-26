# word2vec

## 依赖库说明

### 1. tqdm
用于在命令行界面中创建进度条  
```python
from tqdm import tqdm
import time

items = range(10)
for item in tqdm(items, desc="Test", total=len(items)):
	time.sleep(1)
```
iterable:是一个可迭代对象
desc：进度条前的描述性信息
total：可迭代对象的长度

##### 在pytorch中使用tqdm
在train函数中使用tqdm，讲dataloader做为一个可迭代对象传入tqdm
```python
loop = tqdm((dataloader_train), desc=f"Epoch: [{epoch}/20]", total=len(dataloader_train))
    for img, label in loop:
        img = img.to(device)
        label = label.to(device)
        output = model(img)
        optimizer.zero_grad()
        loss = criterion(output,label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        correct += (torch.argmax(output,dim=1) == label).sum().item()
        loop.set_postfix(loss=loss.item() / label.shape[0])
    print("epoch: {i}    Train Loss: {loss}".format(i=epoch, loss=train_loss))
    print("epoch: {i}    Train Accuracy: {acc}".format(i=epoch, acc=correct / len(dataset_train)))
```

### 2. DataLoader
DataLoader加载数据集必须是torch.utils.data.Dataset的子集。
继承Dataset必须实现两个方法：```__len__ 和 __getitem__```

##### DataLoader的参数
- dataset:Dataset类，PyTorch已有的数据读取接口，决定数据从哪里读取及如何读取；
- batch_size：批大小；默认1
- num_works:是否多进程读取数据；默认0使用主进程来导入数据。大于0则多进程导入数据，加快数据导入速度
- shuffle：每个epoch是否乱序；默认False。输入数据的顺序打乱，是为了使数据更有独立性，但如果数据是有序列特征的，就不要设置成True了。一般shuffle训练集即可。
- drop_last:当样本数不能被batchsize整除时，是否舍弃最后一批数据；
- collate_fn:将得到的数据整理成一个batch。默认设置是False。如果设置成True，系统会在返回前会将张量数据（Tensors）复制到CUDA内存中。
- batch_sampler，批量采样，和batch_size、shuffle等参数是互斥的，一般采用默认None。batch_sampler，但每次返回的是一批数据的索引（注意：不是数据），应该是每次输入网络的数据是随机采样模式，这样能使数据更具有独立性质。所以，它和一捆一捆按顺序输入，数据洗牌，数据采样，等模式是不兼容的。
- sampler，默认False。根据定义的策略从数据集中采样输入。如果定义采样规则，则洗牌（shuffle）设置必须为False。
- pin_memory，内存寄存，默认为False。在数据返回前，是否将数据复制到CUDA内存中。
- timeout，是用来设置数据读取的超时时间的，但超过这个时间还没读取到数据的话就会报错。
- worker_init_fn（数据类型 callable），子进程导入模式，默认为Noun。在数据导入前和步长结束后，根据工作子进程的ID逐个按顺序导入数据。

注意 collate_fn 可以指定读取批量函数方法。

### 3. SummaryWriter
将条目直接写入 log_dir 中的事件文件以供 TensorBoard 使用。
类是torch.utils.tensorboard.SummaryWriter。
在给定目录中创建事件文件，并向其中添加摘要和事件。 该类异步更新文件内容。 这允许训练程序调用方法以直接从训练循环将数据添加到文件中，而不会减慢训练速度。

##### 初始化
第一个参数 log_dir : 用以保存summary的位置
第二个参数是加一些comment
##### 常用函数add_scalar()
函数参数
- tag：要求是一个string，用以描述 该标量数据图的 标题
- scalar_value ：可以简单理解为一个y轴值的列表
- global_step：可以简单理解为一个x轴值的列表，与y轴的值相对应

##### 可视化展示
类似：tensorboard --logdir=ZCH_Tensorboard_Trying_logs --port=6666
这个logdir就是初始化第一个参数。

