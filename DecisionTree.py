import math
import copy
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import drawTree

# 把数据集分为训练集和测试集
def dataSetSplit():
    data = pd.read_csv("income_dataset.csv", index_col=False)
    x, y = data.iloc[:, :], data.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    x_train.to_csv("training.csv", index=False)
    x_test.to_csv("testing.csv", index=False)

# 离散化数值型属性
def stratifyEdu(edu):
    n = len(edu)
    for i in range(n):
        education = int(edu[i])
        if education <= 4:
            edu[i] = 'primary school'
        elif education <= 8:
            edu[i] = 'junior high school'
        elif education <= 12:
            edu[i] = 'senior high school'
        else:
            edu[i] = 'bachelor or above'


def stratifyAge(age):
    n = len(age)
    for i in range(n):
        AGE = int(age[i])
        if AGE <= 30:
            age[i] = 'youth'
        elif AGE <= 60:
            age[i] = 'midlife'
        else:
            age[i] = 'elder'


def stratifyHour(hour):
    n = len(hour)
    for i in range(n):
        HOUR = int(hour[i])
        if HOUR < 30:
            hour[i] = '<30hours'
        elif HOUR <= 60:
            hour[i] = '30~60hours'
        else:
            hour[i] = '>60hours'


# 数据预处理
def dataPreprocess():
    print("数据预处理中...")
    trainingDataSet = []
    testingDataSet = []
    headers = []
    feature = []


    # 处理训练集
    with open('training.csv', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        m = len(headers[:-1])
        for i in range(m):
            feature.append(set())
        ageIdx = headers.index('age')
        eduIdx = headers.index('education')
        hourIdx = headers.index('hours-per-week')
        age = []
        edu = []
        hour = []
        for row in reader:
            age.append(row[ageIdx])
            edu.append(row[eduIdx])
            hour.append(row[hourIdx])
            trainingDataSet.append(row)
        stratifyAge(age)
        stratifyEdu(edu)
        stratifyHour(hour)
        trainLength=len(trainingDataSet)
        for i in range(trainLength):
            trainingDataSet[i][ageIdx] = age[i]
            trainingDataSet[i][eduIdx] = edu[i]
            trainingDataSet[i][hourIdx] = hour[i]
            for j in range(m):
                feature[j].add(trainingDataSet[i][j])

    # 处理测试集
    with open('testing.csv', encoding='utf-8') as ff:
        reader = csv.reader(ff)
        headers = next(reader)
        ageIdx = headers.index('age')
        eduIdx = headers.index('education')
        hourIdx = headers.index('hours-per-week')
        age = []
        edu = []
        hour = []
        for row in reader:
            age.append(row[ageIdx])
            edu.append(row[eduIdx])
            hour.append(row[hourIdx])
            testingDataSet.append(row)
        stratifyAge(age)
        stratifyEdu(edu)
        stratifyHour(hour)
        testLength=len(testingDataSet)
        for i in range(testLength):
            testingDataSet[i][ageIdx] = age[i]
            testingDataSet[i][eduIdx] = edu[i]
            testingDataSet[i][hourIdx] = hour[i]
    # for i in range(m):
    #     featureSet = set([example[i] for example in trainingDataSet])
    #     feature.append(featureSet)
    print("数据预处理完成")
    print("属性:",headers[:-1])
    print()
    # print(feature)
    # print()
    return trainingDataSet, testingDataSet, headers[:-1],feature


# # 训练集
# def generateTrainingData(dataSet,tags):
#     # dataSet = [['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
#     #            ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
#     #            ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
#     #            ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '是'],
#     #            ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '是'],
#     #            ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '否'],
#     #            ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '否'],
#     #            ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '否'],
#     #            ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '否'],
#     #            ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '否']]
#     # tags = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']  # 特征
#
#     feature = []
#     m = len(tags)
#     for i in range(m):
#         featureSet = set([example[i] for example in dataSet])
#         feature.append(featureSet)
#     # print(feature)
#     # print()
#     return dataSet, tags, feature
#
#
# # 测试集
# def generateTestData():
#     dataSet = [['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
#                ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
#                ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '是'],
#                ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '否'],
#                ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '否'],
#                ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '否'],
#                ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '否']]
#     return dataSet


# 计算信息熵
def getInfoEntropy(dataSet):
    n=len(dataSet)
    labelCount={}
    for example in dataSet:
        label=example[-1]
        if label not in labelCount:
            labelCount[label]=1
        else:
            labelCount[label]+=1
    infoEntropy=0.0
    for label in labelCount:
        P=labelCount[label]/n
        infoEntropy -= P * math.log2(P)
    return infoEntropy


# 划分数据集
def splitDataSet(dataSet,featureIdx,featureValue):
    newDataSet=[]
    if len(dataSet) == 0:
        return newDataSet
    for example in dataSet:
        if example[featureIdx] == featureValue:
            newVector=example[:featureIdx]
            newVector.extend(example[featureIdx+1:])
            newDataSet.append(newVector)
    # print("划分数据集:")
    # for item in newDataSet:
    #     print(item)
    # print()
    return newDataSet


# 选择最佳属性
def selectFeature(dataSet,feature):
    m=len(dataSet[0])-1 # 属性的个数
    # print("当前属性个数:",m)
    infoEntropy=getInfoEntropy(dataSet)
    # print("信息熵:",infoEntropy)
    bestInfoGain=0.0
    bestFeatureIdx=-1
    for i in range(m):
        currentFeatureValueSet=feature[i]
        # print("候选属性分支集:",currentFeatureValueSet)
        condEntropy=0.0
        for value in currentFeatureValueSet:
            # print("候选属性分支:",value)
            subDataSet = splitDataSet(dataSet,i,value)
            P = len(subDataSet)/len(dataSet) #在第i个属性取value的概率
            condEntropy += P * getInfoEntropy(subDataSet)
        # print("条件熵:",condEntropy)
        infoGain = infoEntropy - condEntropy
        # print("信息增益:",infoGain)
        # print()
        if infoGain >= bestInfoGain:
            bestInfoGain=infoGain
            bestFeatureIdx=i
    # print("最大信息增益:",bestInfoGain)
    return bestFeatureIdx


# 出现次数最多的属性
def majorFeature(labelList):
    labelCount={}
    for label in labelList:
        if label not in labelCount:
            labelCount[label] = 1
        else:
            labelCount[label] += 1
    labelCount=sorted(labelCount.items(),key=lambda x:x[1],reverse=True)
    # print("分类:",labelCount[0][0])
    # print()
    return labelCount[0][0]


# 建树
def createTree(dataSet,tags,feature,fatherDataSet=None):
    if len(dataSet)==0: # 如果样本为空，设置为叶节点，类别设置为父节点所含样本最多的类别
        labelList=[example[-1] for example in fatherDataSet]
        return majorFeature(labelList)
    labelList=[example[-1] for example in dataSet]
    if len(set(labelList)) == 1: # 只有一种类别
        # print("分类:",labelList[0][0])
        # print()
        return labelList[0]
    if len(dataSet[0]) == 1: # 遍历完了属性
        return majorFeature(labelList)
    bestFeatureIdx=selectFeature(dataSet,feature) # 数据集中属性的下标
    bestFeatureLabel=tags[bestFeatureIdx]
    # print("所选属性节点:",bestFeatureLabel)
    # print()
    DTree={bestFeatureLabel:{}}
    # print("子树:",DTree)
    # print()
    del tags[bestFeatureIdx] # 删除已经选择的属性
    featureValueSet=feature[bestFeatureIdx]
    del feature[bestFeatureIdx]
    # print("当前节点:{},候选属性分支集:{}".format(bestFeatureLabel,featureValueSet))
    for value in featureValueSet:
        # print("当前节点:{},候选属性分支:{}".format(bestFeatureLabel,value))
        subTags=copy.deepcopy(tags)
        subFeature=copy.deepcopy(feature)
        DTree[bestFeatureLabel][value]=createTree(splitDataSet(dataSet,bestFeatureIdx,value),subTags,subFeature,dataSet)
    return DTree


# 分类
def classify(DTree,tags,testFeatureVector):
    root_key= list(DTree.keys())[0] # 根节点名
    featureIdx = tags.index(root_key) # 根节点对应的属性下标
    root_value=DTree[root_key]  # 根节点值（字典）
    for value in root_value:
        if testFeatureVector[featureIdx]==value:
            if type(root_value[value]).__name__ != 'dict': # 子节点是叶节点
                return root_value[value]
            else: # 非叶节点：递归
                return classify(root_value[value],tags,testFeatureVector)


# 测试模型
def testModel(DTree,tags,testDataSet):
    count=0
    for testData in testDataSet:
        if testData[-1] == classify(DTree,tags,testData):
            count+=1
    print("模型准确率")
    print(count/len(testDataSet))


def main():
    # 数据集分割
    dataSetSplit()
    # 数据预处理
    trainingDataSet,testingDataSet,tags,feature=dataPreprocess()
    # # 训练集生成
    # feature = generateTrainingData(trainingDataSet)
    # 建树
    tags1 = copy.deepcopy(tags)
    DTree = createTree(trainingDataSet, tags1, feature)
    print("决策树")
    print(DTree)
    print()
    # 绘制模型
    drawTree.plot_model(DTree, "myTree.gv")
    # # 测试集生成
    # testingDataSet = generateTestData()
    # 测试模型
    testModel(DTree, tags, testingDataSet)


if __name__ == "__main__":
    main()

