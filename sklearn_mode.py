# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 11:34:32 2017

@author: lenovo
"""

import os  
import time  
#from sklearn import metrics  
import numpy as np  
import scipy.io as sio  
import sklearn

# 测试样本对应的真实格点坐标
def test_real(path):
    real1=[]
    for i in range(grid_num):
        for j in range(grid_test_sample):
            real1.append(i+1)
    real=np.array(real1)
    return real
   
def read_data(path):  
#模型：一共有56个格点，每个格点上训练数据100组，测试数据10组
#1. 读取训练数据组，  
#matlab文件名  
    pathtrain=os.path.join(path,'train.mat')  
    traindataset=sio.loadmat(pathtrain).get('data')#训练数据
    pathtest=os.path.join(path,'test.mat')
    testdataset=sio.loadmat(pathtest).get('X_test')#测试数据
    trainlabelset1=[]
    for grid in range(grid_num):
        for sample in range(grid_train_sample):
    #        local_tmp=local[grid]
            local_tmp=grid+1
            trainlabelset1.append(local_tmp)
    trainlabelset = np.array(trainlabelset1)#训练数据集对应坐标
    testlabelset = test_real(path)#测试数据集对应坐标
    return traindataset,trainlabelset,testdataset,testlabelset

## Multinomial Naive Bayes Classifier  
#def naive_bayes_classifier(train_x, train_y):  
#    from sklearn.naive_bayes import MultinomialNB  
#    model = MultinomialNB(alpha=0.01)  
#    model.fit(train_x, train_y)  
#    return model  
#  
  
# KNN Classifier  
# KNeighborsClassifier(n_neighbors=5, weights=’uniform’, algorithm=’auto’, leaf_size=30, p=2,
#                      metric=’minkowski’, metric_params=None, n_jobs=1, **kwargs)
def knn_classifier(train_x, train_y):  
    from sklearn.neighbors import KNeighborsClassifier  
    model = KNeighborsClassifier()  
    model.fit(train_x, train_y)  
    return model  
  
  
# Logistic Regression Classifier  
# LogisticRegression(penalty=’l2’, dual=False, tol=0.0001, C=1.0, fit_intercept=True,
#                   intercept_scaling=1, class_weight=None, random_state=None, solver=’liblinear’,
#                   max_iter=100, multi_class=’ovr’, verbose=0, warm_start=False, n_jobs=1)
def logistic_regression_classifier(train_x, train_y):  
    from sklearn.linear_model import LogisticRegression  
    model = LogisticRegression(penalty='l2')  
    model.fit(train_x, train_y)  
    return model  
  
  
# Random Forest Classifier  
def random_forest_classifier(train_x, train_y):  
    from sklearn.ensemble import RandomForestClassifier  
    model = RandomForestClassifier(n_estimators=8)  
    model.fit(train_x, train_y)  
    return model  
  
  
# Decision Tree Classifier  
def decision_tree_classifier(train_x, train_y):  
    from sklearn import tree  
    model = tree.DecisionTreeClassifier()  
    model.fit(train_x, train_y)  
    return model  
  
  
# GBDT(Gradient Boosting Decision Tree) Classifier  
def gradient_boosting_classifier(train_x, train_y):  
    from sklearn.ensemble import GradientBoostingClassifier  
    model = GradientBoostingClassifier(n_estimators=200)  
    model.fit(train_x, train_y)  
    return model  
  
  
# SVM Classifier  
def svm_classifier(train_x, train_y):  
    from sklearn.svm import SVC  
    model = SVC(kernel='rbf', probability=True)  
    model.fit(train_x, train_y)  
    return model  
  
# SVM Classifier using cross validation  
def svm_cross_validation(train_x, train_y):  
    from sklearn.grid_search import GridSearchCV  
    from sklearn.svm import SVC  
    model = SVC(kernel='rbf', probability=True)  
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}  
    grid_search = GridSearchCV(model, param_grid, n_jobs = 1, verbose=1)  
    grid_search.fit(train_x, train_y)  
    best_parameters = grid_search.best_estimator_.get_params()  
    for para, val in best_parameters.items():  
        print para, val  
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)  
    model.fit(train_x, train_y)  
    return model  
    
def func_rmse(predict_label,real):
    pathlocal=os.path.join(path,'dist1.mat')
    local=sio.loadmat(pathlocal).get('dist')#56个格点坐标
    tmp = predict_label-np.ones(np.shape(predict_label))
    predict_2d = local[map(int,tmp)]
    tmp_real = real-np.ones(np.shape(predict_label))
    real_2d = local[map(int,tmp_real)]
    norm_sqrt = (np.linalg.norm(predict_2d-real_2d,axis=-1))**2
    rmse = np.sqrt(sum(norm_sqrt)/np.shape(predict_label))
    return rmse
    
def get_probability(data_chuang):
    """
    对窗数据来说，维度为：len_chuang*算法个数
    计算每种算法下格点序号出现次数及概率
    """
    mygrid = range(1,57)
    cishu_total=[]
    for sf in range(len_sf):
        cishu=[]
        for item in mygrid:
            cishu_sf = data_chuang[:,sf].tolist().count(item)
            cishu.append(cishu_sf)
        cishu = np.array(cishu)
        cishu_total.append(cishu)
    probio_total = [item/float(len_chuang) for item in cishu_total]
    probio_total = np.array(probio_total).transpose()
    return probio_total
    
    
if __name__ == '__main__':  
    path=os.path.abspath('.')
    thresh = 0.5  
    model_save_file = None  
    model_save = {}  
    grid_num = 56
    grid_train_sample = 100
    grid_test_sample = 10
      
    test_classifiers = [ 'KNN', 'LR', 'RF', 'DT', 'SVM']  
    classifiers = {#'NB':naive_bayes_classifier,   
                  'KNN':knn_classifier,  
                   'LR':logistic_regression_classifier,  
                   'RF':random_forest_classifier,  
                   'DT':decision_tree_classifier,  
                  'SVM':svm_classifier,  
                'SVMCV':svm_cross_validation,  
                 'GBDT':gradient_boosting_classifier,
    }  
      
    print 'reading training and testing data...'  
    train_x, train_y, test_x, test_y = read_data(path)  
    num_train, num_feat = train_x.shape  
    num_test, num_feat = test_x.shape  
    is_binary_class = (len(np.unique(train_y)) == 56)  
    print '******************** Data Info *********************'  
    print '#training data: %d, #testing_data: %d, dimension: %d' % (num_train, num_test, num_feat)  
    
    predict_total2=[]
    rmse_total=[]
    for classifier in test_classifiers:  
        print '******************* %s ********************' % classifier  
        start_time = time.time()  
        model = classifiers[classifier](train_x, train_y)  
        print 'training took %fs!' % (time.time() - start_time)  
        predict = model.predict(test_x)  
        if model_save_file != None:  
            model_save[classifier] = model  
        if is_binary_class:  
#            precision = metrics.precision_score(test_y, predict)  
#            recall = metrics.recall_score(test_y, predict)  
#            print 'precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall)  
            tmp = func_rmse(predict,test_y)
            print tmp
            rmse_total.append(tmp)
        accuracy = sklearn.metrics.accuracy_score(test_y, predict)  
        print 'accuracy: %.2f%%' % (100 * accuracy)   
        predict_total2.append(predict)
    predict_total1 = np.array(predict_total2)
    predict_total = predict_total1.transpose()
#    if model_save_file != None:  
#        pickle.dump(model_save, open(model_save_file, 'wb')) 
    
    """
    对已经被机器学习算法进行预测之后的数据矩阵predict_total
    维度为:测试样本数*算法数
    滑窗对数据集进行处理，窗长度为len_chuang，每次下滑一格
    对窗长度内的数据进行概率计算并DS融合
    """
    len_chuang = 10 # 窗数据长度
    pre_ds=[]
    pre_dsjq=[]
    pre_max=[]
    pre_test=[]
    for start in range(0,560-len_chuang+1):# 滑动窗口
        data_chuang = predict_total[start:len_chuang+start,:]# 取出窗长度的数据    
        len_sf = len(data_chuang[0]) # 多算法个数
        probio_total = get_probability(data_chuang) # 窗数据矩阵中每种算法对应所有格点出现概率
        
        """
        1.普通ds
         等同于求多种算法概率平均之后，概率最大的格点为所预测格点
        """
        init_mean = np.mean(probio_total,1) # 每个格点下多种算法提供的平均概率
        init_ds = init_mean
        # 这部分并没有起到实际作用
        for i in range(len_sf-1):
            init_ds = np.multiply(init_ds,init_mean)
            k = sum(init_ds)
            bel = init_ds/float(k)
        pre_grid = np.argmax(bel)+1
        pre_ds.append(pre_grid)
        """
        2.加强ds
         求得多种算法的源可信度，得到估计
        """
        m_between = np.zeros([len_sf,len_sf])
        for i in range(len_sf):
            for j in range(len_sf):
                m_between[i,j]=sum(np.multiply(probio_total[:,i],probio_total[:,j]))
                # m_between两个证据的内积=两个证据中，每个出现事件的概率乘积*（出现事件的交集/并集）
                # 对单个事件而不是集合事件而言，等同于对应事件的概率乘积之和
        d = np.zeros([len_sf,len_sf])
        sim = np.zeros([len_sf,len_sf])
        for i in range(len_sf):
            for j in range(len_sf):
                d[i,j]=np.sqrt(0.5*(m_between[i,i]+m_between[j,j]-2*m_between[i,j]))
                # d为两个证据间的距离，距离越小表示两个证据提出的意见越一致
                sim[i,j]=1-d[i,j]
                # sim为两个证据之间的相似度，越大代表两个证据之间的一致性越强
        sup = np.zeros(len_sf)
        for i in range(len_sf):
            sup[i]=sum(sim[i,:])-sim[i,i]
            # sup为对每个证据的支持度，为两个证据之间的相似度之和减去该证据自己对自己的支持度
            # 证据对自己的支持度为1
        crd = np.zeros(len_sf)
        for i in range(len_sf):
            crd[i]=float(sup[i])/sum(sup)
            # crd为证据的可信度
            # 即为归一化的支持度，其他证据对该证据支持度越高，则可信度越高
        A = np.zeros(grid_num)
        for i in range(grid_num):
            A[i] = sum(np.multiply(probio_total[i,:],crd))
            # 将可信度作为源权重，估计所有情况下数据出现的概率
        AA = A
        
        # 这部分并没有起到实际作用
        # 对于所有元素均为0-1的概率值，进行元素对于相乘之后并不改变元素的大小排序
        for i in range(len_sf-1):
            init_ds = np.multiply(AA,A)
            # 分子为与某事件有交集的事件概率之乘积
            k = sum(init_ds)
            # 分母K=∑(所有有交集的事件的概率乘积)
            # 或者为1-∑(所有不相交的时间概率乘积)
            # 对全部都是单事件而不是集体事件而言，有交集的事件即为事件其本身
            # K表示了证据的冲突程度，冲突越大，越接近0，一致性越大，越接近1
            bel = init_ds/float(k)
        pre_grid = np.argmax(bel)+1
        pre_dsjq.append(pre_grid)
        # 下面两行代码是为了验证多次融合A矩阵并不能起到结果
        testgrid = np.where(A == np.max(A))[0][0]
        pre_test.append(testgrid+1)
        """
        3.使用窗数据中出现次数最多的作为正确结果
        """
        data_chuang_list = data_chuang.flatten().tolist()
        matrix_cishu = max(data_chuang_list,key=data_chuang_list.count)
        pre_max.append(matrix_cishu)
        
    truth_chuang = test_y[len_chuang-1:]
    rmse_ds = func_rmse(pre_ds,truth_chuang)
    rmse_max = func_rmse(pre_max,truth_chuang)
    rmse_dsjq = func_rmse(pre_dsjq,truth_chuang)
    rmse_test = func_rmse(pre_test,truth_chuang)
    
