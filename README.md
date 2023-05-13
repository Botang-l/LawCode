# LawCode 內容介紹

## 部署方式
- Clone this repo
    ```shell
    $ https://github.com/Botang-l/flask_sever.git
    $ cd flask_sever
    ```
- Install the Python dependencies
    ```shell
    $ pip install -r deployment/requirements.txt
    ```
- Run the server on port 5000
    ```shell
    $ python local_server.py
    ```

## 專案架構
```
LawCode/
├── result/                        # 資料處理後的成果
│   ├── data.json                  # 符合條件的判決資料全文
│   └── path.json                  # 符合條件的判決資料字號
│   └── total_info.json            # 符合條件的判決資料完整資訊
│   └── result.csv                 # 經過特徵篩選後的結果
├── data
│   └── <年份資料請自行抓取>
├── deployment/                     # configuration for deployment
│   └── requirements.txt            # Python package list
├── README.md
└── test.py                         # 使用範例
└── tool.py                         # 工具
```

## 使用說明

1. 資料取得:
    - 到[司法院資料開放平臺](https://opendata.judicial.gov.tw/)取得資料

        ![](https://i.imgur.com/0MVCwlE.png)

2. 使用 LoadData 函數讀取資料
    - 參數
    `noSummaryCourt` : 布林值，表示是否要過濾掉簡易庭的案件，預設為 False。 
    `noDistrictCourt` : 布林值，表示是否要過濾掉地方法院的案件，預設為 False。
    `noHighCourt` : 布林值，表示是否要過濾掉高等法院的案件，預設為 False。
    `noSupremeCount` : 布林值，表示是否要過濾掉最高法院的案件， 預設為 False。
    `YearFrom` : 整數，表示要篩選的案件年分下限，預設為 0。
    `YearTo` : 整數，表示要篩選的案件年分上限，預設為 10000。 
    `Jcase` : 字串，表示要篩選的案件類型，預設為 None。 
    `Jtitle` : 字串，表示要篩選的案件標題，預設為 None。
    `OnlyJudge` : 布林值，表示是否只要保留判決案件， 預設為 False。
    `OnlyOrder` : 布林值，表示是否只要保留裁定案件， 預設為 False。
    
    - 產出結果
    `./result/data.json` : 符合使用者篩選條件的判決資料全文
    `./result/path.json` : 符合使用者篩選條件的判決資料字號
    `./result/total_info.json` : 符合使用者篩選條件的判決資料完整資訊

3. 使用 AnalysisData 函數抓取資料特徵
    - 參數
    `Cond` : 你想要抓取的資料特徵，多個特徵間以頓號分隔。

    - 範例
        ```python=
        analysis_data('裁定,臺灣新北地方法院')
        ```

    - 產出結果
    `result.csv` : 經過特徵篩選後產生的feature。
        > feature 中，判賠比率欄位微程式碼自動產生，若為空值表示程式碼未自動抓取判決結果。
4. 使用 MachineLearning 函數進行機器學習
    - 無參數
    - 產出結果 : 各機器學習使用率
        ```
        LR 準確率: 57.979 %
        LDA 準確率: 57.979 %
        KNN 準確率: 44.681 %
        RFC 準確率: 57.979 %
        SVM 準確率: 57.979 %
        ```
    


