import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd


df = pd.read_csv("4dbe5667-7b6b-41d7-82af-211562424d9a_a0d71eebebaa5ba00b5d1af1dd96a3dd.csv")

df.info(), df.head()

df_cleaned = df.copy()

df_cleaned['CompanyRegistrationdate_date'] = pd.to_datetime(df_cleaned['CompanyRegistrationdate_date'], errors='coerce')

df_cleaned.dropna(inplace=True)

df_cleaned['CompanyStatus'] = df_cleaned['CompanyStatus'].str.strip().str.title()
df_cleaned['CompanyClass'] = df_cleaned['CompanyClass'].str.strip().str.title()


plt.figure(figsize=(10, 5))
df_cleaned['CompanyStateCode'].value_counts().head(10).plot(kind='bar', color='skyblue')
plt.title('Top 10 States with Most Companies')
plt.ylabel('Number of Companies')
plt.xlabel('State Code')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(df_cleaned['AuthorizedCapital'], bins=50, kde=True, color='green')
plt.title('Distribution of Authorized Capital')
plt.xlabel('Authorized Capital')
plt.ylabel('Frequency')
plt.show()

class_counts = df_cleaned['CompanyClass'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('Set2'))
plt.title('Company Class Distribution')
plt.axis('equal')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='CompanyStatus', y='PaidupCapital', data=df_cleaned)
plt.title('Paid-up Capital by Company Status')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8, 6))
corr = df_cleaned[['AuthorizedCapital', 'PaidupCapital', 'nic_code']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

X = df_cleaned[['AuthorizedCapital']]
y = df_cleaned['PaidupCapital']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.xlabel('Actual Paid-up Capital')
plt.ylabel('Predicted Paid-up Capital')
plt.title('Linear Regression: Actual vs Predicted')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.show()

mse, r2
