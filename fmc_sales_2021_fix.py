import pandas as pd
data = pd.read_csv('')
data.info()
df = pd.DataFrame(data)
print(df)

data_deskripsi = df.describe()
print("\nStatistik Deskriptif Data:")
print(data_deskripsi)

import pandas as pd
import re
df['Barang'] = df['Barang'].fillna('').astype(str).str.strip().str.lower()

sapaan = {'pak', 'bapak', 'bu', 'ibu', 'kak', 'mba', 'mbak', 'mas', 'bpk', '-'}
def bersihkan_nama(nama):
    nama = re.sub(r'[^\w\s]', '', str(nama).lower())
    tokens = nama.split()
    tokens_bersih = [word for word in tokens if word not in sapaan]
    return ' '.join(tokens_bersih)
df['Nama Pelanggan'] = df['Nama Pelanggan'].apply(bersihkan_nama)

duplicates = df[df.duplicated()]
duplicate_count = df.duplicated().sum()
print(f'Jumlah data duplikat: {duplicate_count}')
print(duplicates)

df_cleaned = df.drop_duplicates()
duplicates = df_cleaned[df_cleaned.duplicated()]
duplicate_count = df_cleaned.duplicated().sum()
print(f'Jumlah data duplikat: {duplicate_count}')
df_cleaned.shape

df_cleaned = df_cleaned.drop(columns=['Status Pelanggan', 'Supplier', 'Customer Service','Bulan','Tanggal'])
df_cleaned.info()

df_cleaned.isnull()
missingvalue_count = df_cleaned.isnull().sum()
print(f'Jumlah missing value:\n {missingvalue_count}')
df_cleaned = df_cleaned.dropna()

df_cleaned['Kuantitas'] = pd.to_numeric(df_cleaned['Kuantitas'], errors='coerce')
df_cleaned = df_cleaned.dropna(subset=['Kuantitas'])

df_cleaned['Kuantitas'] = df_cleaned['Kuantitas'].astype(int)
df_cleaned['Pendapatan'] = df_cleaned['Pendapatan'].replace({'Rp': '', ',': ''}, regex=True)
df_cleaned['Pendapatan'] = pd.to_numeric(df_cleaned['Pendapatan'], errors='coerce')
print(df_cleaned.dtypes)
df_cleaned.shape

fmc_f = df_cleaned.groupby(by=['Barang'], as_index=False)['Jumlah'].sum()
fmc_f.columns = ['Barang', 'Frequency']
fmc_m = df_cleaned.groupby(by=['Barang'], as_index=False)['Penjualan '].sum()
fmc_m.columns = ['Barang', 'Monetary']
fmc_c = df_cleaned.groupby(['Barang'])['Dari'].nunique().reset_index()
fmc_c.columns = ['Barang', 'Customer Variety']
fmc = fmc_f.merge(fmc_m, on='Barang').merge(fmc_c, on='Barang')
print(fmc)

statistik_deskriptif = fmc[['Frequency', 'Monetary', 'Customer Variety']].describe()
print("\nStatistik Deskriptif:")
print(statistik_deskriptif)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(13, 13))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(fmc['Frequency'], fmc['Monetary'], fmc['Customer Variety'], c='blue', marker='o',  s=175)
ax.set_xlabel('Frekuensi')
ax.set_ylabel('Monetary')
ax.set_zlabel('Customer Variety')
ax.set_title('Scatter Plot 3D Segmentasi Produk')
plt.show()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
columns_to_normalize = ['Frequency', 'Monetary', 'Customer Variety']
fmc[columns_to_normalize] = scaler.fit_transform(fmc[columns_to_normalize])
print("\nNormalisasi FMC penjualan 2021:")
print(fmc)

statistik_deskriptif = fmc[['Frequency', 'Monetary', 'Customer Variety']].describe()
print("\nStatistik Deskriptif:")
print(statistik_deskriptif)

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

optimal_k = 2
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(fmc[columns_to_normalize])
fmc['KMeans_Cluster'] = kmeans.fit_predict(fmc[columns_to_normalize])
ahc = AgglomerativeClustering(n_clusters=optimal_k)
fmc['AHC_Cluster'] = ahc.fit_predict(fmc[columns_to_normalize])
print(fmc)
print(fmc['KMeans_Cluster'].value_counts())
print(fmc['AHC_Cluster'].value_counts())
import plotly.express as px

data = pd.DataFrame(fmc, columns=['Frequency', 'Monetary', 'Customer Variety','KMeans_Cluster'])

# Plotting 3D scatter menggunakan plotly
fig = px.scatter_3d(
    data,
    x='Frequency',
    y='Monetary',
    z='Customer Variety',
    color='KMeans_Cluster',
    title='Scatter Plot 3D Hasil KMeans Clustering'
)

fig.show()

data = pd.DataFrame(fmc, columns=['Frequency', 'Monetary', 'Customer Variety','KMeans_Cluster', 'AHC_Cluster'])

# Plotting 3D scatter menggunakan plotly
fig = px.scatter_3d(
    data,
    x='Frequency',
    y='Monetary',
    z='Customer Variety',
    color='AHC_Cluster',
    title='Scatter Plot 3D Hasil AHC Clustering'
)

fig.show()

from mpl_toolkits.mplot3d import Axes3D

# Plotting
fig = plt.figure(figsize=(13, 13))
ax = fig.add_subplot(111, projection='3d')
colors = ['r', 'b', 'y', 'c', 'm']

# Menggunakan kolom 'KMeans_Cluster' untuk visualisasi cluster
for cluster_label, color in zip(fmc['KMeans_Cluster'].unique(), colors):
    cluster_data = fmc[fmc['KMeans_Cluster'] == cluster_label]
    ax.scatter(
        cluster_data['Frequency'],
        cluster_data['Monetary'],
        cluster_data['Customer Variety'],
        c=color,
        label=f'Cluster {cluster_label}',
        s=175  # Ukuran marker
    )

# Menambahkan label untuk sumbu
ax.set_xlabel('Frequency')
ax.set_ylabel('Monetary')
ax.set_zlabel('Customer Variety')

# Menambahkan judul dan legenda
plt.title('Scatter Plot 3D Hasil Clustering Berdasarkan K-Means')
plt.legend()
plt.show()

# Plotting
fig = plt.figure(figsize=(15, 13))
ax = fig.add_subplot(111, projection='3d')
colors = ['r', 'b', 'y', 'c', 'm']

# Menggunakan kolom 'AHC_Cluster' untuk visualisasi cluster
for cluster_label, color in zip(data['AHC_Cluster'].unique(), colors):
    cluster_data = data[data['AHC_Cluster'] == cluster_label]
    ax.scatter(
        cluster_data['Frequency'],
        cluster_data['Monetary'],
        cluster_data['Customer Variety'],
        c=color,
        label=f'Cluster {cluster_label}',
        s=175
    )

# Menambahkan label untuk sumbu
ax.set_xlabel('Frequency')
ax.set_ylabel('Monetary')
ax.set_zlabel('Customer Variety')

# Menambahkan judul dan legenda
plt.title('Scatter Plot 3D Hasil Clustering Berdasarkan AHC')
plt.legend()
plt.show()

cluster_averages_KMeans = fmc.groupby('KMeans_Cluster')[['Frequency', 'Monetary', 'Customer Variety']].mean()
cluster_averages_AHC = fmc.groupby('AHC_Cluster')[['Frequency', 'Monetary', 'Customer Variety']].mean()
print("Rata-rata nilai Frequency, Monetary, dan Customer Variety untuk tiap cluster dengan K-Means:")
print(cluster_averages_KMeans)
print("Rata-rata nilai Frequency, Monetary, dan Customer Variety untuk tiap cluster dengan AHC:")
print(cluster_averages_AHC)
