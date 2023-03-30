import os
import glob 
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def CountFilter( path, noSummaryCourt, noDistrictCourt, noHighCourt, noSupremeCount):
    
    if(noSummaryCourt and ('簡易庭' in path)):
        return(True)
    if(noDistrictCourt and ('地方法院' in path)):
        return(True)
    if(noHighCourt and ('高等法院' in path)):
        return(True)
    if(noSupremeCount and ('最高法院' in path)):
        return(True)
    return(False)

def TypeFilter(file, YearFrom,YearTo, Jcase, Jtitle):
    
    if(YearFrom > int(file['JYEAR'])):
        return(True)
    if(YearTo < int(file['JYEAR'])):
        return(True)
    if(Jcase != None):
        Jcase = Jcase.split(',')
        if file['JCASE'] not in Jcase:
            return(True)
    if(Jtitle != None):
        Jtitle = Jtitle.split(',')
        if file['JTITLE'] not in Jtitle:
            return(True) 
    return(False)  

def LoadData(noSummaryCourt = False, noDistrictCourt = False,
    noHighCourt = False, noSupremeCount = False, YearFrom = 0,
    YearTo = 10000, Jcase = None, Jtitle = None):
    # 篩選條件
    
    # 當前檔案目錄
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, '*\*\*\*json')
    # data 資料夾路徑
    files = glob.glob(data_dir)
    print("共有",len(files),"個 json 檔")
    words = []
    titles = []
    lists = []
    # 遍歷data資料夾中的所有檔案
    for file in files:
        if(CountFilter(file, noSummaryCourt, noDistrictCourt,
            noHighCourt, noSupremeCount)):
            continue
        # 檢查是否為檔案而非目錄
        if os.path.isfile(file):
            #print('hi')
            # 讀取檔案內容
            with open(file, 'r',encoding='utf-8') as f:
                
                content = json.load(f)
                if(TypeFilter(content, YearFrom, YearTo, Jcase, Jtitle)):
                    continue
                else:
                    content["JFULL"] = content["JFULL"].replace(' ','').replace('\u3000','').replace('\n','').replace('\r','')
                    lists.append(content)
                    words.append(content["JFULL"])
                    titles.append(content["JID"])

    print('其中有',len(titles),'個檔案符合您的篩選條件，相關輸出以放置於 result 資料夾')
    with open('result/data.json','w',encoding='utf-8') as f:
        json.dump(words, f, ensure_ascii=False)
    with open('result/path.json','w',encoding='utf-8') as f:
        json.dump(titles, f, ensure_ascii=False)
    with open('result/total_info.json','w',encoding='utf-8') as f:
        json.dump(lists, f, ensure_ascii=False)


def analysis_data(cond):
    with open('result/total_info.json','r',encoding='utf-8') as f:
        contents = json.load(f)
    feature = cond.split(',')
    data = []
    index = []
    for i in contents:
        index.append(i['JID'])
        data.append([(j in i['JFULL'])for j in feature])
    
    df = pd.DataFrame(data, columns=feature, index=index ,dtype=float)
    df.to_csv('result/result.csv') 
    print('搜尋成功')


def MachineLearning():
    
    # 資料前處理
    data = pd.read_csv('result/result.csv',sep=',')
    data.drop(columns=data.columns[0], axis=1, inplace=True)
    columns = list(data.columns)
    feature = columns[:-1]
    label = columns[-1]
    X_train, y_train, X_test, y_test = train_test_split(
    data[feature], data[label], test_size=0.3, random_state=0)

    # 標準化數據
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    y_train_std = sc.transform(y_train)

    # LR
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(X_train_std, X_test)
    print('LR 準確率:',round(lr.score(y_train_std, y_test)*100,3),'%')

    # LDA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_std, X_test)
    print('LDA 準確率:',round(lda.score(y_train_std, y_test)*100,3),'%')

    # KNN
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier()
    knn.fit(X_train_std, X_test)
    print('KNN 準確率:',round(knn.score(y_train_std, y_test)*100,3),'%')

    # 隨機森林模型
    from sklearn.ensemble import RandomForestClassifier
    rfc=RandomForestClassifier()
    rfc.fit(X_train_std, X_test)
    print('RFC 準確率:',round(rfc.score(y_train_std, y_test)*100,3),'%')

    # SVM
    from sklearn.svm import LinearSVC
    svm=LinearSVC()
    svm.fit(X_train_std, X_test)
    print('SVM 準確率:',round(svm.score(y_train_std, y_test)*100,3),'%')