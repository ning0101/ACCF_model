# ACCF model

### Description 

- 進行模型的訓練與繪製實驗結果圖片



### model

- MLP: 單純neural network
- A_MLP: 使用attention_mechanism
- C_MLP: 結合user-based CF 與 item-based CF



### Highlight 

**資料前處理:**

* 不同檔案的讀取 ex: json, csv, xml
* 時間格式的處理 ex: 2021-06-06, 2021/6/6

**資料標準化:**

* 針對不同的檔案，寫不同的讀取方式(reader)，以放入SensorThingsAPI模型中
* 使用PostgreSQL存取資料



### Tools 

* python
* numpy
* pandas
* scikit-learn
* tensorflow
