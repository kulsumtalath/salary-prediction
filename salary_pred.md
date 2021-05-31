```python
import pandas as pd
```


```python
data=pd.read_csv("salary.csv")
```


```python
data.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>experience</th>
      <th>test_score</th>
      <th>interview_score</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>8.00</td>
      <td>9</td>
      <td>50000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>8.00</td>
      <td>6</td>
      <td>45000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>five</td>
      <td>6.00</td>
      <td>7</td>
      <td>60000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>two</td>
      <td>10.00</td>
      <td>10</td>
      <td>65000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>seven</td>
      <td>9.00</td>
      <td>6</td>
      <td>70000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>three</td>
      <td>7.00</td>
      <td>10</td>
      <td>62000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ten</td>
      <td>7.85</td>
      <td>7</td>
      <td>72000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>eleven</td>
      <td>7.00</td>
      <td>8</td>
      <td>80000</td>
    </tr>
  </tbody>
</table>
</div>




```python
dict={"five":5,"two":2,"seven":7,"three":3,"ten":10,"eleven":11}
```


```python
dict
```




    {'five': 5, 'two': 2, 'seven': 7, 'three': 3, 'ten': 10, 'eleven': 11}




```python
data.isnull().sum()
```




    experience         0
    test_score         0
    interview_score    0
    salary             0
    dtype: int64




```python
def convert_to_int(word):
    word_dict={"one":1,"five":5,"two":2,"seven":7,"three":3,"ten":10,"eleven":11,"0":0}
    return word_dict[word]
```


```python
X=data.iloc[:,:3]
```


```python
y=data.iloc[:,-1]
```


```python
X['experience']=X['experience'].apply(lambda x:convert_to_int(x))
```


```python
X
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>experience</th>
      <th>test_score</th>
      <th>interview_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>8.00</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>8.00</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>6.00</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>10.00</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>9.00</td>
      <td>6</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3</td>
      <td>7.00</td>
      <td>10</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10</td>
      <td>7.85</td>
      <td>7</td>
    </tr>
    <tr>
      <th>7</th>
      <td>11</td>
      <td>7.00</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
```


```python
regressor.fit(X,y)
```




    LinearRegression()




```python
import pickle
```


```python
pickle.dump(regressor,open('model.pkl','wb'))
```


```python
model=pickle.load(open('model.pkl','rb'))
print(model.predict([[0,10,10]]))
```

    [58340.74887092]
    


```python

```
