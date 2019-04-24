import jieba
import numpy as np
import time
import jieba.posseg as psg
import re
import math

class textR(object):
    def __init__(self):
        self.window = 5 #默认 窗口
        self.maxIter = 100 #默认 最大迭代次数
        self.topK = 5  #默认 关键字数量
        self.d = 1   #默认 阻尼系数
        # 词性要求， 默认任意词性都可做关键词，实际词性为名词类如('ns', 'n', 'vn', 'v')作关键词
        self.posList = []

    def get_stopword_list(self):
        # 停用词表存储路径，每一行为一个词，按行读取进行加载
        # 进行编码转换确保匹配准确率
        stop_word_path = './stopword.txt'
        stopword_list = [sw.replace('\n', '') for sw in open(stop_word_path,encoding="utf8").readlines()]
        return stopword_list

    def keyword(self,text):
        begin = time.time()
        stopword_list = self.get_stopword_list()
        # words = [w for w in jieba.cut(text) if w not in stopword_list]
        # 去停用词， 分词，词性标注
        Word_pos_mat = np.matrix([[w, v] for w, v in psg.cut(text) if w not in stopword_list])
        words, pos = np.array(Word_pos_mat[:, 0].T)[0], np.array(Word_pos_mat[:, 1].T)[0]
        uni_words = set(words)  #去重
        uni_words_size = len(uni_words)
        wsize = len(words)
        wscore = {k:1 for k in words}  #得分
        if uni_words_size < self.window:  #若文本小于窗口，则直接返回分词结果
            return wscore
        w_id = {v: k for k, v in enumerate(uni_words)}  #{w0:0,...} 映射表
        w_id2 = {k:v for k, v in enumerate(uni_words)}  # {0:w0,...} 映射表2
        w_matrix = np.zeros(shape=[uni_words_size, uni_words_size])   #words二维矩阵
        for i in range(self.window,wsize+1):
            co_words = words[i - self.window:i]
            for wj in co_words:
                for wk in co_words:
                    if wj != wk:
                        if w_matrix[w_id[wj],w_id[wk]]==0:
                            w_matrix[w_id[wj], w_id[wk]] = 1
                        if w_matrix[w_id[wk], w_id[wj]] == 0:
                            w_matrix[w_id[wk], w_id[wj]] = 1
        oldS = [wscore[w_id2[i]] for i in range(uni_words_size)]
        outNum = []
        for i in range(0, uni_words_size):
            outNum.append(sum(w_matrix[i,:]))   #每个词语的 出链数
        for iter in range(0,self.maxIter):
            for i in range(0,uni_words_size):
                #计算每个词语分出去权重
                w_matrix[i, :] = w_matrix[i, :] > 0
                w_matrix[i,:] = w_matrix[i, :]*wscore[w_id2[i]]/outNum[i]
            for w in wscore:
                # 计算每个词语得分
                wscore[w] = (1-self.d)+self.d*sum(w_matrix[:,w_id[w]])
            newS = [wscore[w_id2[i]] for i in range(uni_words_size)]
            equal= [0 if abs(k-v)<1e-6 else 1 for k, v in zip(oldS, newS)] #收敛精度 1e-6
            equal_num = sum(equal)
            oldS = newS
            print("第"+str(iter)+"次迭代结果,未收敛个数--->",str(equal_num)+"   cost time--->"+ str(time.time()-begin))
            if equal_num == 0:
                break
            # print("第" + str(iter) + "次迭代结果", str(sum(newS)), sep="--->")
        if len(self.posList) > 0 :
            # 按词性过滤，去重
            for k in list(set([words[index] for index in range(0,wsize) if pos[index] not in self.posList])):
                wscore.pop(k)
            print(len(wscore))
        return sorted(wscore.items(),key = lambda item:item[1],reverse=True)[:self.topK]

    def fenJu(self,textFile):
        text = ''
        with open(textFile, 'r', encoding="utf8") as f:
            for line in f.readlines():
                text += line.strip()
        sentences = re.split('。|！|\!|\.|？|\?', text)  # 保留分割符
        if len(sentences[-1]) == 0:
            del sentences[-1]  # 删除最后空句子
        sentences = list(set(sentences))  # 去重
        return sentences

    def similarity(self,sent1, sent2):
        stopword_list = self.get_stopword_list()
        words1 = [w for w in jieba.cut(sent1) if w not in stopword_list]
        words2 = [w for w in jieba.cut(sent2) if w not in stopword_list]

        co_words = [w for w in words1 if w in words2]  # 共同词

        sim = len(co_words) * 1.0 / (math.log(len(words1), 2) + math.log(len(words2), 2))
        return sim

    def abstract(self,sentences):
        begin = time.time()
        sentences_size = len(sentences)
        sent_score = {sid: 1 for sid in range(sentences_size)}
        sent_matrix = np.zeros(shape=[sentences_size, sentences_size])
        for si, sent1 in enumerate(sentences):
            for sj, sent2 in enumerate(sentences):
                temp_sim = self.similarity(sent1, sent2)
                sent_matrix[si, sj] = temp_sim
                sent_matrix[sj, si] = temp_sim
        outNum = []
        for i in range(sentences_size):
            outNum.append(sent_matrix[i, :] / sum(sent_matrix[i, :]))  # 每个句子的 权重
        oldS = [sent_score[i] for i in range(sentences_size)]
        for iter in range(0, self.maxIter):
            for i in range(sentences_size):
                sent_matrix[i, :] = sent_score[i] * outNum[i]  # 每个句子的 出链分值
            for i in range(sentences_size):
                sent_score[i] = sum(sent_matrix[:, i])  # 计算每个句子的得分
            newS = [sent_score[i] for i in range(sentences_size)]

            equal = [0 if abs(k - v) < 1e-6 else 1 for k, v in zip(oldS, newS)]  # 收敛精度 1e-6
            equal_num = sum(equal)
            oldS = newS
            print("第" + str(iter) + "次迭代结果,未收敛个数--->", str(equal_num) + "   cost time--->" + str(time.time() - begin))
            if equal_num == 0:
                break
            sortSent_temp = sorted(sent_score.items(), key=lambda item: item[1], reverse=True)[:self.topK]
            sortSent = [sentences[item[0]] for item in sortSent_temp]
        return sortSent

if __name__ == '__main__':
    corpus_path = './data/test'
    text = ''
    with open(corpus_path, 'r',encoding="utf8") as f:
        for line in f.readlines():
            text +=line.strip()
    print(text)
    textR = textR()
    textR.topK = 10
    textR.posList = ['ns', 'n', 'vn', 'v']
    keywords= textR.keyword(text)
    print(keywords)
    print("--------------分界线-------------------")
    textR.topK = 3
    print(textR.abstract(textR.fenJu(corpus_path)))


