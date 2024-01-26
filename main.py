# Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
from torch import nn
import torch.optim as optim
# 语句中的*通配符表示导入模块中的所有内容，包括所有函数、类、变量等。这意味着您可以直接使用模块中的所有符号，而无需使用模块名称作为前缀，比如make_data()的使用获取data。
from datasets import *
from transformer import Transformer

if __name__ == "__main__":

    enc_inputs, dec_inputs, dec_outputs = make_data()
    # batch_size->2: 这是指定的批处理大小，即每次加载的样本数目。
    # shuffle->True: 这是指定是否在每个epoch开始时对数据进行随机打乱。
    # so 整个代码的作用是创建一个数据加载器，使用自定义数据集MyDataSet，每次加载2个样本,两句话。这意味着每个训练步骤中，模型将会处理两个样本。且在每个epoch开始时进行数据打乱。这样可以方便地用于模型的训练过程。
    loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)

    # Transformer(): 创建了一个Transformer模型的实例，包括，会根据输入字典中元素个数和维度设置Encode过程中的生成qkv的权重矩阵Wq Wk Wv，以及Decode过程中taget输入生成qkv的对应权重矩阵的维度。
    # 在深度学习中，模型的计算可以在不同的设备上进行，包括CPU和GPU。 .to('cpu'): 这部分将创建的Transformer模型移动到CPU设备上计算。如果是NV的cuda就使用Transformer().cuda()
    model = Transformer().to('cpu')
    # 损失函数：交叉熵函数（为什么交叉熵函数可以作为模型训练的损失函数）
    # 用于指定要忽略的类别的索引。在训练过程中，可能存在一些样本的目标类别是无效的，设置 ignore_index 可以忽略这些无效的类别，使其不影响损失的计算，忽略 占位符 索引为0.
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 随机梯度下降（Stochastic Gradient Descent）优化器，SGD是一种常用的优化算法，用于最小化损失函数，通过不断迭代模型参数来更新模型。
    # model.parameters() ：获取模型中的所有可学习参数。
    # lr学习率（learning rate）：表示每次更新时参数沿着梯度的反方向移动的步长大小。
    # 动量 momentum：动量是一种在更新过程中考虑之前梯度的方法，有助于加速训练过程并增加模型稳定性。momentum参数表示动量的权重，一般设置为接近1的值。
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

    for epoch in range(20):

        for enc_inputs, dec_inputs, dec_outputs in loader:  # enc_inputs : [batch_size, src_len] 上面在使用加载器的时候设置了batch_size :2 ,并在初始化模型的时候设置了对应两句话的权重矩阵维度信息
                                                            # dec_inputs : [batch_size, tgt_len]
                                                            # dec_outputs: [batch_size, tgt_len]

            # to 方法将张量 enc_inputs dec_inputs dec_outputs移动到 CPU 设备。
            # .to('cpu') 是 PyTorch 中的一个函数，允许你明确指定张量所在的设备。
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.to('cpu'), dec_inputs.to('cpu'), dec_outputs.to('cpu')

            # 执行 Transformer 模型的前向传播，并获取模型输出以 及注意力权重的操作。
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)

            # 方法dec_outputs.view(-1) 是将模型的dec_outputs张量进行扁平化，从而计算交叉熵损失。
            loss = criterion(outputs, dec_outputs.view(-1))

            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

            # 这个函数的作用是将优化器中所有参数的梯度归零？
            # 在训练模型时，梯度需要累积，但在每个批次或迭代之前，你通常会使用这个函数来清零梯度。
            # 这是因为 PyTorch 默认会对梯度进行累加，而你希望每个迭代都使用新的梯度值。
            # 简单的说就是进来一个 batch 的数据，计算一次梯度，更新一次网络
            optimizer.zero_grad()

            # 这个函数用于执行反向传播，计算损失相对于模型参数的梯度。
            # 它会沿着计算图向后传播误差，并为每个参数计算梯度。
            loss.backward()

            # 这个函数用于执行优化步骤，即使用梯度下降等优化算法更新模型参数。
            # 在调用了 loss.backward() 后，梯度信息已经被计算，而 optimizer.step() 则根据这些梯度更新模型的参数，
            # 使模型逐渐收敛到损失函数的最小值。
            optimizer.step()


    print("打印模型...")
    print(model)

    # 将整个PyTorch模型保存到名为 'model.pth' 的文件中，包括模型的架构和参数（也可以选择只保存模型参数）。
    torch.save(model, 'model.pth')
    print("保存模型...")