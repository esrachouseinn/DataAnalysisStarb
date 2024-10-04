import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Veri setini yükleme
sns.set_style("darkgrid")
df = pd.read_csv("starbucks.csv", index_col=0)

# Veri hakkında genel bilgiler
print(df.head())
print(df.info())
print(df.describe())
print("Ürün türleri:", df["type"].unique())
print("Toplam ürün sayısı:", df["item"].nunique())
print("Ürün türlerine göre dağılım:")
print(df.groupby("type")["item"].count())

# Grafikleri tek bir sekmede toplama
plt.figure(figsize=(15, 20))

# Kalori Değerleri Bar Grafiği
plt.subplot(3, 2, 1)
sns.barplot(x="type", y="calories", data=df)
plt.title("Kalori Değerleri")

# Protein Değerleri Bar Grafiği
plt.subplot(3, 2, 2)
sns.barplot(x="type", y="protein", data=df, palette="Set2")
plt.title("Protein Değerleri")

# Karbonhidrat Değerleri Bar Grafiği
plt.subplot(3, 2, 3)
sns.barplot(x="type", y="carb", data=df, palette="Set2")
plt.title("Karbonhidrat Değerleri")

# Lif Değerleri Bar Grafiği
plt.subplot(3, 2, 4)
sns.barplot(x="type", y="fiber", data=df, palette="Set2")
plt.title("Lif Değerleri")

# Kalori Dağılım Grafiği
plt.subplot(3, 2, 5)
sns.histplot(x="calories", data=df, color="red", kde=True)
plt.title("Kalori Grafiği")

# Protein Dağılım Grafiği
plt.subplot(3, 2, 6)
sns.histplot(x="protein", data=df, color="green", kde=True)
plt.title("Protein Grafiği")

plt.tight_layout()

# Makine öğrenmesi kısmı: Karar ağacı oluşturma
x = df[["calories", "fat", "carb", "fiber", "protein"]]
y = df["type"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# Tahmin ve doğruluk oranı
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Doğruluk:", accuracy)

# Örnek bir veri ile tahmin yapma
prediction = model.predict(pd.DataFrame([[250, 2, 40, 7, 9]], columns=["calories", "fat", "carb", "fiber", "protein"]))
print("Tahmin edilen ürün tipi:", prediction)

# Karar ağacının görselleştirilmesi
plt.figure(figsize=(15, 10))
plot_tree(model, feature_names=x.columns, class_names=model.classes_, filled=True)
plt.title("Karar Ağacı")
plt.tight_layout()

# Tüm grafikleri aynı anda gösterme
plt.show()
