import os
import glob 
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import re
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

s='\d*'
s0='萬'
s1='\d+'
s2='[ ]*元'
s5=','
s6='\w'
s8='聲明[:：]*'
s9='[\D+]*[(\d)]*[：:]*[.、()]*[\d.]*\D+'
s10='[被]告'
s11='[共應連帶]*'
s12='給[付原告(（下同）)]*'
s7=s10+s11+s12
s13='[.\d]*'
s14='訴請判令[:：]*'
s15='[\w（）\(\),、\<\>\[\]\-\. 「」:：％\w]*'
s16='[一二三四五六七八九零壹貳參肆伍陸柒捌玖]'
s17='求為判[決:：]*'
s18='[(（下同）)]*'
s19='[億,萬千\d+]*'
s20='總計'
#s13='[\w（）\(\),、\<\>\[\]\-\.「」:：％]'
g1='判決如下：'
#訴請判令：被告應給付7,190,144元

def CountFilter(path, noSummaryCourt, noDistrictCourt, noHighCourt, noSupremeCount):
    
    if(noSummaryCourt and ('簡易庭' in path)):
        return(True)
    if(noDistrictCourt and ('地方法院' in path)):
        return(True)
    if(noHighCourt and ('高等法院' in path)):
        return(True)
    if(noSupremeCount and ('最高法院' in path)):
        return(True)
    return(False)

def TypeFilter(file, YearFrom,YearTo, Jtitle):
    
    if(YearFrom > int(file['JYEAR'])):
        return(True)
    if(YearTo < int(file['JYEAR'])):
        return(True)
    if(Jtitle != None):
        Jtitle = Jtitle.split(',')
        if file['JTITLE'] not in Jtitle:
            return(True) 
    return(False)  

def LoadData(noSummaryCourt = False, noDistrictCourt = False,
    noHighCourt = False, noSupremeCount = False, YearFrom = 0,
    YearTo = 10000, Jcase = None, Jtitle = None, 
    OnlyJudge = False, OnlyOrder = False):
    # 篩選條件
    
    # 當前檔案目錄
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, 'data\*\*\*json')
    # data 資料夾路徑
    files = glob.glob(data_dir)
    print("共有",len(files),"個 json 檔")
    words = []
    titles = []
    lists = []
    # 遍歷data資料夾中的所有檔案
    for file in tqdm(files):
        if(CountFilter(file, noSummaryCourt, noDistrictCourt,
            noHighCourt, noSupremeCount)):
            continue
        # 檢查是否為檔案而非目錄
        if os.path.isfile(file):
            case = file.split(',')[2]
            
            if (case == Jcase):
                with open(file, 'r',encoding='utf-8') as f:
                    content = json.load(f)
                    if(TypeFilter(content, YearFrom, YearTo, Jtitle)):
                        continue
                    else:
                        content["JFULL"] = content["JFULL"].replace(' ','').replace('\u3000','').replace('\n','').replace('\r','').replace('、','').replace('.','').replace('(','').replace(')','')
                        try:
                            end = re.search('號',content["JFULL"]).start()
                        except:
                            continue
                        name = content["JFULL"][:end+1]
                        content["name"] = name
                        if((OnlyJudge and ("判決" not in name)) or (OnlyOrder and ("裁定" not in name))):
                            continue                     
                        lists.append(content)
                        words.append(content["JFULL"])
                        titles.append(name)

    print('其中有',len(titles),'個檔案符合您的篩選條件，相關輸出以放置於 result 資料夾')
    with open('result/data.json','w',encoding='utf-8') as f:
        json.dump(words, f, ensure_ascii=False)
    with open('result/path.json','w',encoding='utf-8') as f:
        json.dump(titles, f, ensure_ascii=False)
    with open('result/total_info.json','w',encoding='utf-8') as f:
        json.dump(lists, f, ensure_ascii=False)

def LoadCEData(OnlyJudge=False, OnlyOrder=False):
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, 'data\*\*\*json')
    # data 資料夾路徑
    files = glob.glob(data_dir)
    print("共有",len(files),"個 json 檔")
    words = []
    titles = []
    lists = []

    for file in tqdm(files):
        if os.path.isfile(file):
            Jcase = file.split(',')[2]
            if Jcase == "建":
                with open(file, 'r',encoding='utf-8') as f:
                    content = json.load(f)
                    content["JFULL"] = content["JFULL"].replace(' ','').replace('\u3000','').replace('\n','').replace('\r','').replace('、','').replace('.','').replace('(','').replace(')','')
                    try:
                        end = re.search('號',content["JFULL"]).start()
                    except:
                        continue
                    name = content["JFULL"][:end+1]
                    if((OnlyJudge and ("判決" not in name)) or (OnlyOrder and ("裁定" not in name))):
                        continue 
                    content["name"] = name                    
                    lists.append(content)
                    words.append(content["JFULL"])
                    titles.append(name)

    print('其中有',len(titles),'個檔案符合您的篩選條件，相關輸出以放置於 result 資料夾')
    with open('result/data.json','w',encoding='utf-8') as f:
        json.dump(words, f, ensure_ascii=False)
    with open('result/path.json','w',encoding='utf-8') as f:
        json.dump(titles, f, ensure_ascii=False)
    with open('result/total_info.json','w',encoding='utf-8') as f:
        json.dump(lists, f, ensure_ascii=False)


def AnalysisData(Cond, MaxLimit=True):
    with open('result/total_info.json','r',encoding='utf-8') as f:
        contents = json.load(f)
    feature = Cond.split(',')
    data = []
    index = []
    print("正在根據使用者提供之需求分析資料")
    error=0
    for i in tqdm(contents):
        
        index.append(i['name'])
        data.append([(j in i['JFULL'])for j in feature])
        #print('1')
        claimed_amount = get_claimed_amount(i['JFULL'])
        #print('2')
        total_amount = get_total_amount(i['JFULL'])
        try:
            num = min(claimed_amount/total_amount,1) if MaxLimit else claimed_amount/total_amount
            data[-1].append(float(num))
        except:
            error += 1
            data[-1].append(None)

    
    feature.append('判賠比率')        
    df = pd.DataFrame(data, columns=feature, index=index ,dtype=float)
    df.to_csv('result/result.csv',encoding="UTF-8-sig") 
    print('搜尋成功')
    print('失敗比數', error)


def MachineLearning(path='result/result.csv'):
    
    # 資料前處理
    data = pd.read_csv(path,sep=',')
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
    print('LR AUC:',round(roc_auc_score(y_test, lr.predict_proba(y_train_std)[:, 1]),3))
    print()

    # LDA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_std, X_test)
    print('LDA 準確率:',round(lda.score(y_train_std, y_test)*100,3),'%')
    print('LDA AUC:',round(roc_auc_score(y_test, lda.predict_proba(y_train_std)[:, 1]),3))
    print()

    # KNN
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier()
    knn.fit(X_train_std, X_test)
    print('KNN 準確率:',round(knn.score(y_train_std, y_test)*100,3),'%')
    print('KNN AUC:',round(roc_auc_score(y_test, knn.predict_proba(y_train_std)[:, 1]),3))
    print()

    # 隨機森林模型
    from sklearn.ensemble import RandomForestClassifier
    rfc=RandomForestClassifier()
    rfc.fit(X_train_std, X_test)
    print('RFC 準確率:',round(rfc.score(y_train_std, y_test)*100,3),'%')
    print('RFC AUC:',round(roc_auc_score(y_test, rfc.predict_proba(y_train_std)[:, 1]),3))
    print()

    # SVM
    from sklearn.svm import LinearSVC
    svm=LinearSVC()
    svm.fit(X_train_std, X_test)
    print('SVM 準確率:',round(svm.score(y_train_std, y_test)*100,3),'%')
    #print('SVM AUC:',round(roc_auc_score(y_test, svm.predict_proba(y_train_std)[:, 1])))
    print()

def get_claimed_amount(text): 
    try:
        match = re.search('判決如下：主文\w+[億,萬千\d+億,萬千\d+億,萬千\d+億,萬千\d+億,萬千\d+億,萬千\d+億,萬千\d+]*元',text)
        if match:
            word = match.group()
            s = word.index('幣')
            e = word.index('元')
            word = word[s+1:e]
            return(chinese_to_number(word))
        else:
            return 0
    except:
        return 0
    #return(word)

def chinese_to_number(a):
    a = a.replace(',','')
    b=0
    a=a.replace('','參').replace('','參')
    a=a.replace('零','0').replace('壹','1').replace('貳','2').replace('參','3').replace('肆','4').replace('伍','5').replace('陸','6')
    a=a.replace('柒','7').replace('捌','8').replace('玖','9').replace('拾','0!').replace('佰','00!').replace('仟','000!').replace('萬','!乘萬!').replace('億','!乘萬!').replace('兆','!乘萬!')
    a=a.replace('一','1').replace('二','2').replace('三','3').replace('四','4').replace('五','5').replace('六','6')
    a=a.replace('七','7').replace('八','8').replace('九','9').replace('十','0!').replace('百','00!').replace('千','000!')
    a=a.split('!')
    for i in range(0,len(a)):
        if(a[i]==''):
            continue
        if(a[i]=='乘萬'):
            b=b*10000
        elif(i==0 and int(a[i])<1):
            b+=10**len(a[i])
        else:
            c = int(a[i])
            b+=c
    return(b)

def get_total_amount(text):
    patterns = [
    s8+s15+s7+s15+s16+s15+s0+s15+s2,
    s8+s9+s7+s9+s1+s19+s19+s19+s19+s19+s19+s19+s19+s13+s2,
    s14+s9+s7+s9+s1+s19+s19+s19+s19+s19+s19+s19+s19+s13+s2,
    s14+s9+s7+s9+s1+s19+s19+s19+s19+s19+s19+s19+s19+s13+s2,
    s17+s9+s7+s9+s1+s19+s19+s19+s19+s19+s19+s19+s19+s13+s2,
    s20+s9+s7+s9+s1+s19+s19+s19+s19+s19+s19+s19+s19+s13+s2
    ]
    try:
        for idx, pattern in enumerate(patterns):
            match = re.search(pattern, text)     
            if match:
                match = re.findall(s1+s19+s19+s19+s19+s19+s19+s19+s19+s13,match.group())[-1] if idx else re.findall(s16+s15+s0+s15+s2,match.group())[-1]
                number = chinese_to_number(match)
                return(number)
        else:
            return
    except:
        return
    
def DeleteNull_and_OneHotEncoding():
    data = pd.read_csv('result/result.csv',sep=',')
    #data = data.drop('裁定',axis=1)
    #data = data.drop('臺灣新北地方法院',axis=1)
    data.columns.values[0] = '判決名稱'
    with open('result/total_info.json','r',encoding='utf-8') as f:
        contents = json.load(f)
    title = []
    for i in contents:
        title.append(i['JTITLE'].replace('等',''))
    data = data.assign(判決案由 = title)   
    with open('result/path.json','r',encoding='utf-8') as f:
        contents = json.load(f)
    year = []
    court = []
    num_except = 0
    for i in contents:
        try:
            y = re.search('\d+年度',i).group().replace('年度','')
        
        except:
            y = re.search('\d+年',i).group().replace('年','')
            num_except += 1

        c = re.search('\w+法院',i).group()
        year.append(y)
        court.append(c)
        
    #print('錯誤數 :', num_except, '以解決 :', 1)
    data = data.assign(法院別 = court)
    data = data.assign(年度 = year)
    data = data.dropna()
    data = data.assign(民法第227條之2_ = lambda x: x['民法第227-2條'] + x['第227條之2']+x['227條第2']+x['二二七之二']+x['二二七條之二']+x['情勢變更']+x['情事變更'])
    data = data.drop(['民法第227-2條','第227條之2', '227條第2', '二二七之二', '二二七條之二', '情勢變更', '情事變更'],axis=1)
    data['民法第227條之2_'] = data['民法第227條之2_'].apply(lambda x: min(x, 1))
    data = data.assign(民法第495條_ = lambda x: x['民法第495條'] + x['第四九五'] + x['四百九十五']) 
    data = data.drop(['民法第495條','第四九五','四百九十五'],axis=1)
   

    # 將需要編碼的欄位取出
    cols_to_encode = ['法院別', '年度', '判決案由']
    data_to_encode = data[cols_to_encode]

    # 創建 OneHotEncoder 對象
    encoder = OneHotEncoder()

    # 使用 OneHotEncoder 對需要編碼的欄位進行 one-hot 編碼
    onehot = encoder.fit_transform(data_to_encode).toarray()

    # 將 one-hot 編碼後的結果轉換成 DataFrame
    onehot_df = pd.DataFrame(onehot, columns=encoder.get_feature_names_out(cols_to_encode))

    # 將編碼後的結果與未編碼的欄位結合起來
    result_df = pd.concat([onehot_df, data.drop(cols_to_encode, axis=1).reset_index(drop=True)], axis=1)
    result_df = result_df.drop(['判決名稱'], axis=1)
    df = result_df['判賠比率'].rank(ascending=False)
    # 創建 result 欄位
    result_df['結果'] = df.apply(lambda x: 1 if x <= len(df)/2 else 0)
    result_df = result_df.drop('判賠比率', axis=1)
    result_df.to_csv('result/result.csv',encoding="UTF-8-sig", index=False) 
 
def data_concatenation(new_feature, origin_features):
    data = pd.read_csv('result/result.csv',sep=',')
    origin_features = origin_features.split('、')
    data[new_feature] = data[origin_features].sum(axis=1)
    data = data.drop(origin_features,axis=1)
    data[new_feature] = data[new_feature].apply(lambda x: min(x, 1))
    data.to_csv('result/result.csv',encoding="UTF-8-sig", index=False) 