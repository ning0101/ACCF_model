# ACCF model

## Model overview
<img src="image/overview.png" alt="Cover" width="70%"/>


### Level1: similarity level model

 <img src="image/l1.jpg" alt="C" width="60%"/>
 
 1. 將rating matrix透過三種不同similarity的計算方式得到similarity matrix。
 2. 將similarity matrix分別input到neural network 與 attention mechanism中，得出所預測的下一個月的similarity matrix。
 3. 透過預測的similarity matrix計算prediction rating matrix，並與真實的rating matrix做比較 。


 
### prediction level model
 <img src="image/l2.jpg" alt="Co" width="60%"/>

### combine level model
 <img src="image/l3.jpg" alt="Cov" width="60%"/>



### Description 

* 進行模型的訓練:
  * 包含 similarity level, prediction level, combine level
  * ACCF model 的參數實驗，包含: learning rate, epoch, batch size, unit, layers
* 實驗結果圖片繪製
* 評價指標: RMSE, MSE, MAE
* baseline 比較: UCF, itemKNN, SVD, ACF




### Tools 

* python
* numpy
* pandas
* scikit-learn
* tensorflow
