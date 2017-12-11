import pandas as pd
import jieba
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import PlaintextCorpusReader
from nltk.classify import accuracy as nltk_accuracy
import random
#读取文件并添加标签
filepath = "test.csv"
data = pd.read_csv(filepath,header=None,encoding='utf8')
data.columns=["label","content"]

spam = data[data["label"] == 1]    #取垃圾短信数据集
normal = data[data["label"] == 0]  #取正常短信数据集

spamlist = []
normallist = []
scontent = spam["content"];      #取垃圾短信内容
ncontent = normal["content"];    #取正常短信内容

for i in range(len(spam)):
    content = scontent.iloc[i].replace("\ue310","") #去除短信中的空格，这里编码为\ue310，不清楚为什么
    content =' '.join(jieba.cut(content))            #用jieba进行分词
    spamlist.append(content)                          #分词保存到list中

for j in range(len(normal)):
    content = ncontent.iloc[j].replace("\ue310","") #去除空格
    content = ' '.join(jieba.cut(content))           #分词
    normallist.append(content)                        #分词保存

dataframe = pd.DataFrame({'spam':spamlist})           #将list用DataFrame，以便保存到CSV文件中
dataframe.to_csv('spam.csv', encoding='utf_8_sig', header=False, index=False) #保存到文件中
dataframe = pd.DataFrame({'normal':normallist})       #将list用DataFrame，以便保存到CSV文件中
dataframe.to_csv('normal.csv', encoding='utf_8_sig', header=False, index=False) #保存到文件中


message_corpus = PlaintextCorpusReader('./',['spam.csv','normal.csv']) #取出分词文件


all_message = message_corpus.words()  #所有分词保存为list

def massage_feature(word,num_letter=1):        #分词特征化
    return {'feature':word[-num_letter:]}
labels_name = ([(massage,'垃圾') for massage in message_corpus.words('spam.csv')]+[(massage,'正常') for massage in message_corpus.words('normal.csv')])  #给特征分类


random.seed(7)
random.shuffle(labels_name)


featuresets = [(massage_feature(n),massage) for (n,massage) in labels_name]  #调整格式
train_set,test_set = featuresets[400:],featuresets[:400]       #取2000个数据前400个为测试后1600个为训练
classifier = NaiveBayesClassifier.train(train_set)             #调用nltk中的NaiveBayesClassifier函数，传参训练
print('结果准确率：',str(100*nltk_accuracy(classifier,test_set))+str('%'))   #传测试集参数并预测准确率




