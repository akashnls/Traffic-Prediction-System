import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
data=pd.read_csv("data.csv")
df=pd.DataFrame(data)
X=df[[ 'Hour', 'DayOfWeek', 'Weather', 'VehicleCount' ]]
Y=df['Traffic']
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=10)
model=DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy*100)


with open("traffic_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Predicted Traffic for new data:")
hour = 14
day_of_week = 3
weather = 2
vehicle_count = 50
new_data = pd.DataFrame({'Hour': [hour], 'DayOfWeek': [day_of_week], 'Weather': [weather], 'VehicleCount': [vehicle_count]})
predicted_traffic = model.predict(new_data)
print(predicted_traffic)

