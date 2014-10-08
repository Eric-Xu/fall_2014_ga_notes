### General Python
Type casting
```python
print float("99")
print int(2.5)
print str(99)
```


### Pandas

Series transformations:
```python
heart_data['sex']=heart_data['sex'].apply(str)

data.Title = data.Title.apply(lambda x: x.split(' (')[0])
```

Create a new categorical column:
```python
# /fall_2014_assignments/lab04/EDA_lab.ipynb
def age_grouping(age):
    if age < 18:
        return "<18"
    elif age >= 18 and age < 65:
        return "18-64"
    elif age > 65:
        return "65+"
    else:
        return np.nan

df['age_group'] = df.Age.apply(lambda x: age_grouping(x))
```

