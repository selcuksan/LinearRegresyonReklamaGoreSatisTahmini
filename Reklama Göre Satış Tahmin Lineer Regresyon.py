import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Advertising.csv")
print(df.columns)
print(df.info)
print(df.describe().T)

cost=df[["TV"]]
sales=df[["sales"]]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(cost,sales,test_size=0.33,random_state=False)

from sklearn.linear_model import LinearRegression
lR = LinearRegression()
lR.fit(x_train,y_train)
predict_Sales = lR.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.scatter(x_train,y_train)
plt.plot(x_test,predict_Sales,"r")

print(predict_Sales)
print(y_test)

plt.title("Reklama Göre Satış Tahmin")
plt.ylabel("Satış Sayısı")
plt.xlabel("TV Harcamaları")
plt.show()


predictDf=pd.DataFrame(lR.predict(cost.head(10)))
result = pd.concat([sales.head(10),predictDf],axis=1)
result.columns=["Gerçek Satis","Tahmin Satis"]

print(result)