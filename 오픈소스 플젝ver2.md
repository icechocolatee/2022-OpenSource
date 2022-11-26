```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import preprocessing
from pandas import DataFrame
```


```python
df = pd.read_csv("PTA data(수정).csv")
```


```python
df_right = df[['ACR_250', 'ACR_500', 'ACR_1000', 'ACR_2000', 'ACR_4000', 'ACR_8000',
               'BCR_250', 'BCR_500', 'BCR_1000', 'BCR_2000', 'BCR_4000',
               'WRS_R_Ientensity', 'WRS_R_Score', 'SRT_R_Ientensity']]

df_left = df[['ACL_250', 'ACL_500', 'ACL_1000', 'ACL_2000', 'ACL_4000', 'ACL_8000',
               'BCL_250', 'BCL_500', 'BCL_1000', 'BCL_2000', 'BCL_4000',
               'WRS_L_Ientensity', 'WRS_L_Score', 'SRT_L_Ientensity']]
```


```python
dataR = df_right.to_numpy()
dataL = df_left.to_numpy()

data = np.concatenate((dataR, dataL))
data.shape
```




    (58050, 14)




```python
X = data[:, :11]
y = data[:, -3]
```


```python
uni, uni_count = np.unique(X, axis=0, return_counts=True)
unis = np.c_[uni, uni_count]
unis_sort = unis[(-1*unis[:, -1]).argsort()]
```


```python
n = 100
unis_data = unis_sort[np.where(unis_sort[:, -1] > n)]
print("X가 중복되는 값 오름차순({}개 이상): {}개".format(n, len(unis_data)), end="\n\n")

for data_ in unis_data:
    print(data_)
```

    X가 중복되는 값 오름차순(100개 이상): 10개
    
    [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0. 956.]
    [100. 100. 100. 100. 100. 100.  65.  80.  80.  80.  80. 826.]
    [  5.   5.   5.   5.   5.   5.   0.   0.   0.   0.   0. 606.]
    [100. 100. 100. 100. 100. 100.   0.   0.   0.   0.   0. 374.]
    [100. 100. 100. 100. 100. 100.  55.  70.  80.  80.  80. 309.]
    [ 10.   5.   5.   5.   5.   5.   0.   0.   0.   0.   0. 308.]
    [ 10.  10.  10.  10.  10.  10.   0.   0.   0.   0.   0. 193.]
    [  5.   5.   5.   5.   5.  10.   0.   0.   0.   0.   0. 171.]
    [ 10.   5.   5.   5.   5.  10.   0.   0.   0.   0.   0. 154.]
    [ 10.  10.   5.   5.   5.   5.   0.   0.   0.   0.   0. 142.]
    


```python
X.shape
```




    (58050, 11)




```python
y.shape
```




    (58050,)




```python
unis.shape
```




    (43781, 12)



## 비정상 데이터 샘플 삭제


```python
idx = [] #삭제할 데이터 인덱스
```


```python
# AC값이 전부 0인 행들의 인덱스 추출
dataACR = data[:, :6]
for i in range(dataACR.shape[0]):
    if (np.all(dataACR[i] == 0)):
        idx.append(i)
len(idx)
```




    1005




```python
# BC값이 전부 0인 행들의 인덱스 추출
dataBCR = data[:, 6:11]
for i in range(dataBCR.shape[0]):
    if (np.all(dataBCR[i] == 0)):
        idx.append(i)
len(idx)
```




    17611




```python
# WRS_I가 0인 행들의 인덱스 추출
dataTarget = data[:, 11]
for i in range(dataTarget.shape[0]):
    if (np.any(dataTarget[i] == 0)):
        idx.append(i)
len(idx)
```




    20890




```python
# 인덱스에 해당하는 행 삭제
data_del = np.delete(data, np.unique(np.array(idx)), axis=0).astype('float32')
data_del = data_del[~np.isnan(data_del).any(axis=1), :]
data_del.shape
```




    (40576, 14)




```python
idx_lsts = []
for data_ in unis_data[:, :-1]:
    idx_tmp = np.where(np.all(data[:, :-3] == data_, axis=1))
    idx_lsts.append(idx_tmp[0])
```


```python
score_name = ["WRS_I", "WRS_S", "SRT_I"]
X_labels = ["dB", "Score", "dB"]
for idx_lst, PTA, total in zip(idx_lsts, unis_data[:,:-1], unis_data[:,-1]):
    print("PTA: ", end="")
    for value in PTA:
        print("{:3d}, ".format(int(value)), end="")
    print()
    
    plt_data = data[idx_lst][:, -3:]
    fig, axs = plt.subplots(1, 3, figsize=(12,5))
    for scores, ax, name, xlabel in zip(plt_data.T, axs, score_name, X_labels):
        ax.hist(scores, 21)
        ax.set_title(name)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Counts")
    plt.subplots_adjust(wspace=0.35)
    plt.show()
    print("Total: {}".format(int(total)))
    print()
```

    PTA:   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 
    


    
![png](output_17_1.png)
    


    Total: 956
    
    PTA: 100, 100, 100, 100, 100, 100,  65,  80,  80,  80,  80, 
    


    
![png](output_17_3.png)
    


    Total: 826
    
    PTA:   5,   5,   5,   5,   5,   5,   0,   0,   0,   0,   0, 
    


    
![png](output_17_5.png)
    


    Total: 606
    
    PTA: 100, 100, 100, 100, 100, 100,   0,   0,   0,   0,   0, 
    


    
![png](output_17_7.png)
    


    Total: 374
    
    PTA: 100, 100, 100, 100, 100, 100,  55,  70,  80,  80,  80, 
    


    
![png](output_17_9.png)
    


    Total: 309
    
    PTA:  10,   5,   5,   5,   5,   5,   0,   0,   0,   0,   0, 
    


    
![png](output_17_11.png)
    


    Total: 308
    
    PTA:  10,  10,  10,  10,  10,  10,   0,   0,   0,   0,   0, 
    


    
![png](output_17_13.png)
    


    Total: 193
    
    PTA:   5,   5,   5,   5,   5,  10,   0,   0,   0,   0,   0, 
    


    
![png](output_17_15.png)
    


    Total: 171
    
    PTA:  10,   5,   5,   5,   5,  10,   0,   0,   0,   0,   0, 
    


    
![png](output_17_17.png)
    


    Total: 154
    
    PTA:  10,  10,   5,   5,   5,   5,   0,   0,   0,   0,   0, 
    


    
![png](output_17_19.png)
    


    Total: 142
    
    

## 중복데이터 2개 이상 삭제


```python
n = 2
unis_data2 = unis_sort[np.where(unis_sort[:, -1] >= n)]
print("X가 중복되는 값 오름차순({}개 이상) : {}개".format\
      (n, len(unis_data2)), end="\n\n")
```

    X가 중복되는 값 오름차순(2개 이상) : 4639개
    
    


```python

```
