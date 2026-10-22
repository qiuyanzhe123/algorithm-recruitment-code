import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("airline-passengers.csv")
print(data.head())
print("数据总行数：",len(data))
plt.plot(data["Passengers"])
plt.xlabel("Month")
plt.ylabel("Passengers (thousands)")
plt.title("Monthly Airline Passengers")
plt.savefig("passengers_original.jpg",dpi=300,bbox_inches="tight")
plt.show()