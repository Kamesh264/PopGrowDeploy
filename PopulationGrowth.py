import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv("Population Growth.csv")
print(df.head())
print(df.info())

y = df['Population Growth Rate']
x = df.drop(columns = ['Population Growth Rate'])

print(x.head())

y.head()

def apply_percent(percentage):
    percentage = percentage.strip('%')  # Remove the percentage symbol
    return  float(percentage)

x['Growth Rate'] = x['Growth Rate'].apply(apply_percent)

print(x.info())

print(x.head())

def get_billion(billion_number):
    split_numbers = billion_number.split(",")
    
    # Join the split numbers without commas
    joined_number = "".join(split_numbers)
    joined_number = int(joined_number)
    return joined_number/1000000

y= y.apply(get_billion)

y.head()
print("x - info")
print(x.info())
print("y - info")
# print(y.info())

from sklearn.model_selection import train_test_split as tts
x_train, x_test, y_train, y_test = tts(x, y, test_size = 0.2, random_state = 1)

from sklearn.ensemble import RandomForestRegressor
model_rfr = RandomForestRegressor()

model_rfr.fit(x_train, y_train)

y_pred = model_rfr.predict(x_test)

from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))


pickle.dump(model_rfr, open("PopulationGrowth.pkl",'wb'))
