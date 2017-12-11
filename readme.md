2017/12/11
目标：实现朴素贝叶斯分类垃圾短信功能
数据：traini.csv为全部80万条数据，test.csv为前2000条（程序中用前两千条的前400条为测试集，后1600条为训练集）
程序说明：
        1.读入test.csv数据，python jieba包进行分词，分词结果存入文件spam.csv，normal.csv中
		2.读取分词文件，调整格式并添加标签。
		3.读取训练集数据到nltk.NaiveBayesClassifier.train中，训练模型
		4.读取测试数据到nltk.classify.accuracy中预测准确率。前2000个数据准确率为81%左右。
未完成：预测没有标签的数据。