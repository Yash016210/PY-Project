import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('winequality-red.csv', sep=';')  # ← FIXED HERE
print(df.head())
print(df.info())

df.drop_duplicates(inplace=True)

df['quality_label'] = df['quality'].apply(lambda q: 'low' if q <= 4 else 'medium' if q <= 6 else 'high')
df.to_csv('cleaned_wine.csv', index=False)

sns.countplot(x='quality', data=df)
plt.title("Wine Quality Distribution")
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()
