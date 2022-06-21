# Machine-Learning-SKlearn
implementing common Machine Learning algorithms with SKlearn 

# prognosis

### Call Constructor with the model path 

```python
myclass = prognosis("/Users/nouromran/Diabetic_Retinopathy")
```

## or 

```python
myclass = prognosis("/Users/nouromran/10-year-risk-of-death")
```


## Diabetic_Retinopathy 


### input ( 4  attributes ) 



```python
input = [ 85.180507 , 120.106129 ,  92.605936 , 125.065534]

         #Age    , Systolic_BP  , Diastolic_BP   ,  Cholesterol
```


## 10-year-risk-of-death


### input ( 18 attributes ) 



```python
arr= [ 45.0,	68.0,	441.0,	1.0,	40.6,	10.0,	4.7	,200.0,	115.0,	1.77,	6.6,	2.0	,104.0,	355.0,	32.4,	5.5,	22.683195,	36.0,]
print(myclass.ten_years_death(arr))
```
