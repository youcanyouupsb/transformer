# Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
import torch
import torch.utils.data as Data

             # Encoder_input  Decoder_input      Decoder_output
sentences = [['我 是 学 生 P', 'S I am a student', 'I am a student E'],         # S: 开始符号
             ['我 喜 欢 学 习', 'S I like learning P', 'I like learning P E'],  # E: 结束符号
             ['我 是 男 生 P', 'S I am a boy', 'I am a boy E']]                 # P: 占位符号，如果当前句子不足固定长度用P占位

src_vocab = {'P': 0, '我': 1, '是': 2, '学': 3, '生': 4, '喜': 5, '欢': 6, '习': 7, '男': 8}  # 词源字典  字：索引
src_idx2word = {src_vocab[key]: key for key in src_vocab}
src_vocab_size = len(src_vocab)  # 字典字的个数
tgt_vocab = {'P': 0, 'S': 1, 'E': 2, 'I': 3, 'am': 4, 'a': 5, 'student': 6, 'like': 7, 'learning': 8, 'boy': 9}
idx2word = {tgt_vocab[key]: key for key in tgt_vocab}   # 把目标字典转换成 索引：字的形式
tgt_vocab_size = len(tgt_vocab)                         # 目标字典尺寸
src_len = len(sentences[0][0].split(" "))               # Encoder输入的最大长度 5
tgt_len = len(sentences[0][1].split(" "))               # Decoder输入输出最大长度 5


# 把sentences 转换成字典索引
def make_data():
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]] # 将sentences中的“我是学生p” 作为key查找到value[1,2,3,4,0]给到enc_input，以此内推，遍历sentences中的“我喜欢学习”...
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]] # 将sentences中的“S I am a student” 作为key查找到value[1,3,4,5,6]给到dec_input，以此内推，遍历sentences中的“S I like learning P”...
        dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]] # 将sentences中的“I am a student E” 作为key查找到value[3,4,5,6,2]dec_output，以此内推，遍历sentences中的“I like learning P E”...
        enc_inputs.extend(enc_input)#将每次循环得到的enc_input 给到enc_inputs，在之前对enc_inputs 进行扩容
        dec_inputs.extend(dec_input)#同上
        dec_outputs.extend(dec_output)#同上
        # torch.LongTensor(enc_inputs)的作用是将enc_inputs转换为PyTorch中的LongTensor数据类型的张量。
        # 具体来说，它将包含整数的列表或数组（enc_inputs）转换为PyTorch张量，其中张量的元素类型为64位整数。这种数据类型主要用于存储整数数据。
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)


# 自定义数据集函数
# torch.utils.data.Dataset是PyTorch中用于自定义数据集的类，
# 用户可以通过继承该类来定义自己的数据集，
# 并在继承时重载__len__()：返回数据集的大小和__getitem__()这两个魔法方法。
class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    # 魔法方法：__len__返回数据集的大小。
    #eg：
    # class CustomDataset(torch.utils.data.Dataset):
    #     def __init__(self, data):
    #         self.data = data
    #
    #     def __len__(self):
    #         return len(self.data)
    #
    # # 使用自定义数据集
    # custom_data = [1, 2, 3, 4, 5]
    # dataset = CustomDataset(custom_data)
    # print(len(dataset))  # 输出: 5
    def __len__(self):
        return self.enc_inputs.shape[0]

    # 魔法方法：__getitem__实现索引数据集中的某一个数据。
    #eg：
    # class CustomDataset(torch.utils.data.Dataset):
    #     def __init__(self, data):
    #         self.data = data
    #
    #     def __len__(self):
    #         return len(self.data)
    #
    #     def __getitem__(self, index):
    #         return self.data[index]
    #
    # # 使用自定义数据集
    # custom_data = [1, 2, 3, 4, 5]
    # dataset = CustomDataset(custom_data)
    # print(dataset[2])  # 输出: 3
    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]
