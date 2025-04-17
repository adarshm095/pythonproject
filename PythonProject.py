import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



df=pd.read_csv("C:/Users/madar/Downloads/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69.csv")
print(df.info())
print(df.describe())

print(df.isnull().sum())

df['pollutant_min'].fillna(df['pollutant_min'].mean(), inplace=True)
df['pollutant_max'].fillna(df['pollutant_max'].mean(), inplace=True)
df['pollutant_avg'].fillna(df['pollutant_avg'].mean(), inplace=True)

print(df.info())
print(df.isnull().sum())

#Convert last_update to datetime
df['last_update'] = pd.to_datetime(df['last_update'], format='%d/%m/%Y %H:%M:%M')



# 1. Pollutant trends over time
df_sorted = df.sort_values('last_update')
sns.lineplot(data=df_sorted)
plt.title('Pollutant Average Over Time')
plt.xlabel("last_update")
plt.ylabel("pollutant_avg")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Distribution of pollutant values
for col in ['pollutant_avg']:
    sns.histplot(df[col], kde=True, bins=30)
    plt.title('Distribution fof pollutant')
    plt.xlabel("pollutant avg")
    plt.ylabel("count")
    plt.show()

# 3. Correlation heatmap
sns.heatmap(df[['pollutant_min', 'pollutant_max', 'pollutant_avg']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Between Pollutant Metrics')
plt.show()

# 4. Comparison across cities
city_avg = df.groupby('city')['pollutant_avg'].mean().sort_values(ascending=False)
sns.barplot(x=city_avg.index, y=city_avg.values)
plt.xticks(rotation=45)
plt.title('Average Pollutant by City')
plt.tight_layout()
plt.show()

# 5. Most Common Pollutants 
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='pollutant_id', order=df['pollutant_id'].value_counts().index)
plt.title('Most Common Pollutants')
plt.xlabel('Pollutants')
plt.ylabel('Count')
plt.show()

# 6. Pollution Levels by State
plt.figure(figsize=(10, 5))
state_avg = df.groupby('state')['pollutant_avg'].mean().sort_values(ascending=False)
sns.barplot(x=state_avg.index, y=state_avg.values)
plt.title('Average Pollution Levels by State')
plt.xticks(rotation=30)
plt.ylabel('Pollutant Averages')
plt.xlabel('State')
plt.show()

 #7. Pollution Intensity by Geographic Location
plt.figure(figsize=(12,5))
sns.scatterplot(data=df, x='longitude', y='latitude', hue='pollutant_avg', palette='coolwarm', size='pollutant_avg')
plt.title('Pollution Intensity by Location')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='Pollutant Avg')
plt.show()



#Future Prediction

# Convert datetime
df['last_update'] = pd.to_datetime(df['last_update'], format='%d-%m-%Y %H:%M:%S')
df['timestamp'] = df['last_update'].astype(np.int64) 


# Define independent and dependent variables
X = df[['timestamp']]
y = df['pollutant_avg']

# Split dataset into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)


# Predict future pollution level
latest_date = df['last_update'].max()
future_date = latest_date + pd.Timedelta(days=7)
future_timestamp = int(future_date.timestamp())

future_pollution = model.predict([[future_timestamp]])[0]
print("\nPredicted Pollution Level 7 Days Later:", future_pollution)

# Plotting the regression
plt.figure(figsize=(10, 5))
plt.scatter(X, y, color='blue', label='Actual Data', alpha=0.5)
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')
plt.xlabel('Timestamp')
plt.ylabel('Pollutant Average')
plt.title('Pollution Trend Over Time (Regression)')
plt.legend()
plt.show()









