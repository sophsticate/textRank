import jieba
import re
import math
import numpy as np
import time


def get_stopword_list():
    # 停用词表存储路径，每一行为一个词，按行读取进行加载
    # 进行编码转换确保匹配准确率
    stop_word_path = './stopword.txt'
    stopword_list = [sw.replace('\n', '') for sw in open(stop_word_path, encoding="utf8").readlines()]
    return stopword_list

def similarity(sent1,sent2):
    stopword_list = get_stopword_list()
    words1 = [w for w in jieba.cut(sent1) if w not in stopword_list]
    words2 = [w for w in jieba.cut(sent2) if w not in stopword_list]

    co_words = [w for w in words1 if w in words2]  #共同词

    sim = len(co_words)*1.0/(math.log(len(words1),2) + math.log(len(words2),2))
    return sim

def fenJu(textFile):
    text = ''
    with open(textFile, 'r', encoding="utf8") as f:
        for line in f.readlines():
            text += line.strip()
    sentences = re.split('。|！|\!|\.|？|\?',text)         # 保留分割符
    if len(sentences[-1]) ==0:
        del sentences[-1]  #删除最后空句子
    sentences = list(set(sentences))  #去重
    return sentences

def abstract(sentences,maxIter,topK):
    begin = time.time()
    sentences_size = len(sentences)
    sent_score = {sid:1 for sid in range(sentences_size)}
    sent_matrix = np.zeros(shape=[sentences_size,sentences_size])
    for si,sent1 in enumerate(sentences):
        for sj,sent2 in enumerate(sentences):
            temp_sim = similarity(sent1,sent2)
            sent_matrix[si, sj] = temp_sim
            sent_matrix[sj, si] = temp_sim
    outNum = []
    for i in range(sentences_size):
        outNum.append(sent_matrix[i,:]/sum(sent_matrix[i,:]))  #每个句子的 权重
    oldS = [sent_score[i] for i in range(sentences_size)]
    for iter in range(0, maxIter):
        for i in range(sentences_size):
            sent_matrix[i, :] = sent_score[i]*outNum[i]  #每个句子的 出链分值
        for i in range(sentences_size):
            sent_score[i] = sum(sent_matrix[:,i]) # 计算每个句子的得分
        newS = [sent_score[i] for i in range(sentences_size)]

        equal = [0 if abs(k - v) < 1e-6 else 1 for k, v in zip(oldS, newS)]  # 收敛精度 1e-6
        equal_num = sum(equal)
        oldS = newS
        print("第" + str(iter) + "次迭代结果,未收敛个数--->", str(equal_num) + "   cost time--->" + str(time.time() - begin))
        if equal_num == 0:
            break
        sortSent_temp = sorted(sent_score.items(), key=lambda item: item[1], reverse=True)[:topK]
        sortSent = [sentences[item[0]] for item in sortSent_temp]
    return sortSent







corpus_path = './data/test'

print(abstract(fenJu(corpus_path),100,3))
