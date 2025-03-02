# %% [markdown]
# # <center>Hotel Booking Demand Dataset Analysis</center>
#
# <center><img src="https://images.unsplash.com/photo-1566073771259-6a8506099945?ixlib=rb-1.2.1&auto=format&fit=crop&w=1000&q=80" width="800"></center>
#
# ## Table of Contents
# 1. [Data Description](#1-data-description)
# 2. [Data Loading](#2-data-loading)
# 3. [Data Understanding](#3-data-understanding)
#    - [Basic Statistics](#31-basic-statistics)
#    - [Missing Values Analysis](#32-missing-values-analysis)
#    - [Data Visualization](#33-data-visualization)
# 4. [Data Preparation](#4-data-preparation)
#    - [Handling Missing Values](#41-handling-missing-values)
#    - [Handling Outliers](#42-handling-outliers)
#    - [Encoding Categorical Variables](#43-encoding-categorical-variables)
#    - [Feature Engineering](#44-feature-engineering)
#    - [Feature Selection](#45-feature-selection)
# 5. [Conclusion](#5-conclusion)

# %% [markdown]
# # 1. Data Description
#
# ## Nama Dataset dan Sumbernya
# Dataset yang digunakan adalah **Hotel Booking Demand** yang tersedia di [Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand).
#
# ## Deskripsi Singkat
# Dataset ini berisi informasi tentang pemesanan hotel dari dua jenis hotel: Resort Hotel dan City Hotel. Data mencakup pemesanan yang tiba antara 1 Juli 2015 dan 31 Agustus 2017, termasuk pemesanan yang berhasil check-in dan pemesanan yang dibatalkan.
#
# Dataset ini sangat berguna untuk analisis perilaku pemesanan hotel, prediksi pembatalan, dan strategi penetapan harga. Informasi yang terkandung di dalamnya dapat membantu hotel untuk:
# - Mengoptimalkan strategi penetapan harga
# - Memprediksi pembatalan pemesanan
# - Memahami preferensi tamu
# - Menganalisis pola musiman dalam pemesanan
#
# ## Jumlah Data
# - **Jumlah Sampel**: Dataset ini berisi lebih dari 119.000 observasi
# - **Jumlah Fitur**: Terdapat 32 kolom/fitur
# - **Label**: Kolom 'is_canceled' dapat digunakan sebagai label untuk model prediksi pembatalan
#
# ## Format Data
# Dataset tersedia dalam format CSV (Comma Separated Values).
#
# ## Deskripsi Kolom
#
# * **hotel**: Jenis hotel (Resort Hotel atau City Hotel)
# * **is_canceled**: Nilai yang menunjukkan apakah pemesanan dibatalkan (1) atau tidak (0)
# * **lead_time**: Jumlah hari antara tanggal pemesanan dan tanggal kedatangan
# * **arrival_date_year**: Tahun tanggal kedatangan
# * **arrival_date_month**: Bulan tanggal kedatangan
# * **arrival_date_week_number**: Nomor minggu dari tanggal kedatangan
# * **arrival_date_day_of_month**: Hari dalam bulan dari tanggal kedatangan
# * **stays_in_weekend_nights**: Jumlah malam akhir pekan (Sabtu atau Minggu) yang dipesan
# * **stays_in_week_nights**: Jumlah malam hari kerja (Senin sampai Jumat) yang dipesan
# * **adults**: Jumlah orang dewasa
# * **children**: Jumlah anak-anak
# * **babies**: Jumlah bayi
# * **meal**: Jenis paket makanan yang dipesan
# * **country**: Negara asal tamu
# * **market_segment**: Segmen pasar pemesanan
# * **distribution_channel**: Saluran distribusi pemesanan
# * **is_repeated_guest**: Nilai yang menunjukkan apakah tamu adalah tamu berulang (1) atau tidak (0)
# * **previous_cancellations**: Jumlah pembatalan sebelumnya oleh tamu
# * **previous_bookings_not_canceled**: Jumlah pemesanan sebelumnya yang tidak dibatalkan oleh tamu
# * **reserved_room_type**: Kode jenis kamar yang dipesan
# * **assigned_room_type**: Kode jenis kamar yang diberikan
# * **booking_changes**: Jumlah perubahan/amandemen yang dilakukan pada pemesanan
# * **deposit_type**: Jenis deposit yang dibayarkan
# * **agent**: ID agen perjalanan yang melakukan pemesanan
# * **company**: ID perusahaan/entitas yang melakukan pemesanan
# * **days_in_waiting_list**: Jumlah hari pemesanan berada dalam daftar tunggu
# * **customer_type**: Jenis pelanggan
# * **adr**: Average Daily Rate (Tarif Harian Rata-rata)
# * **required_car_parking_spaces**: Jumlah tempat parkir mobil yang diminta
# * **total_of_special_requests**: Jumlah permintaan khusus dari pelanggan
# * **reservation_status**: Status reservasi (Check-Out, Canceled, No-Show)
# * **reservation_status_date**: Tanggal status reservasi terakhir ditetapkan

# %% [markdown]
# # 2. Data Loading
#
# Pada bagian ini, kita akan memuat dataset Hotel Booking Demand ke dalam lingkungan pemrograman Python. Kita akan menggunakan library Pandas untuk memuat dan memanipulasi data.

# %%
# Import library yang diperlukan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import plotly.express as px
import folium
from folium.plugins import HeatMap
import warnings

# Mengabaikan warning untuk tampilan yang lebih bersih
warnings.filterwarnings('ignore')

# Mengatur gaya visualisasi
plt.style.use('fivethirtyeight')
%matplotlib inline

# Mengatur opsi tampilan pandas
pd.set_option('display.max_columns', 32)
pd.set_option('display.max_rows', 50)

# %% [markdown]
# ## Memuat Dataset
#
# Kita akan memuat dataset menggunakan fungsi `read_csv()` dari Pandas. Dataset tersedia dalam format CSV, sehingga mudah dimuat menggunakan Pandas.

# %%
# Memuat dataset
# Catatan: Jika Anda menjalankan kode ini di lingkungan lokal, pastikan path file sudah benar
# Jika menggunakan Kaggle Notebook, gunakan path berikut:
try:
    df = pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')
except:
    # Alternatif path jika path di atas tidak berfungsi
    try:
        df = pd.read_csv('hotel_bookings.csv')
    except:
        print("File tidak ditemukan. Pastikan file CSV berada di lokasi yang benar.")

# Menampilkan 5 baris pertama dari dataset
df.head()

# %% [markdown]
# ## Tantangan dalam Memuat Data dan Solusinya
#
# Beberapa tantangan yang mungkin dihadapi saat memuat dataset ini:
#
# 1. **Path File**: Jika menjalankan kode di lingkungan lokal, pastikan path file sudah benar. Solusinya adalah dengan menggunakan path relatif atau absolut yang sesuai dengan lokasi file.
#
# 2. **Encoding**: Beberapa file CSV mungkin menggunakan encoding yang berbeda. Jika terjadi error saat memuat file, coba tambahkan parameter `encoding` seperti `encoding='utf-8'` atau `encoding='latin1'`.
#
# 3. **Separator**: Meskipun format file adalah CSV (Comma Separated Values), beberapa file mungkin menggunakan separator lain seperti titik koma (;) atau tab. Jika terjadi error, coba tambahkan parameter `sep` seperti `sep=';'` atau `sep='\t'`.
#
# 4. **Missing Values**: Dataset mungkin memiliki nilai yang hilang dengan representasi yang berbeda seperti 'NA', 'N/A', atau string kosong. Pandas secara default mengenali beberapa representasi nilai yang hilang, tetapi jika diperlukan, tambahkan parameter `na_values` seperti `na_values=['NA', 'N/A', '', ' ']`.
#
# Dalam kasus dataset Hotel Booking Demand ini, tidak ada tantangan khusus dalam memuat data karena formatnya standar dan konsisten.

# %% [markdown]
# # 3. Data Understanding
#
# Pada bagian ini, kita akan melakukan eksplorasi awal terhadap dataset untuk memahami karakteristik dan pola yang ada di dalamnya.

# %% [markdown]
# ## 3.1 Basic Statistics
#
# Pertama, kita akan melihat informasi umum tentang dataset seperti jumlah baris dan kolom, tipe data, dan statistik dasar.

# %%
# Melihat informasi umum tentang dataset
print("Informasi Dataset:")
print(f"Jumlah Baris: {df.shape[0]}")
print(f"Jumlah Kolom: {df.shape[1]}")
print("\nTipe Data:")
df.info()

# %%
# Melihat statistik dasar untuk kolom numerik
print("Statistik Dasar untuk Kolom Numerik:")
df.describe()

# %%
# Melihat statistik dasar untuk kolom kategorikal
print("Statistik Dasar untuk Kolom Kategorikal:")
df.describe(include=['object'])

# %%
# Melihat nilai unik untuk setiap kolom kategorikal
for col in df.select_dtypes(include=['object']).columns:
    print(f"\nKolom: {col}")
    print(f"Jumlah Nilai Unik: {df[col].nunique()}")
    print(f"Nilai Unik: {df[col].unique()[:10] if df[col].nunique() > 10 else df[col].unique()}")
    if df[col].nunique() > 10:
        print("...")

# %% [markdown]
# ## 3.2 Missing Values Analysis
#
# Selanjutnya, kita akan menganalisis nilai yang hilang (missing values) dalam dataset.

# %%
# Menghitung jumlah nilai yang hilang untuk setiap kolom
missing_values = pd.DataFrame({
    'Jumlah Nilai Hilang': df.isna().sum(),
    'Persentase Nilai Hilang': (df.isna().sum() / len(df) * 100).round(2)
})

missing_values = missing_values.sort_values('Jumlah Nilai Hilang', ascending=False)
missing_values

# %%
# Visualisasi nilai yang hilang
plt.figure(figsize=(12, 8))
msno.matrix(df)
plt.title('Visualisasi Nilai yang Hilang', fontsize=16)
plt.show()

# %%
# Visualisasi nilai yang hilang dengan heatmap
plt.figure(figsize=(12, 8))
msno.heatmap(df)
plt.title('Heatmap Nilai yang Hilang', fontsize=16)
plt.show()

# %% [markdown]
# ## 3.3 Data Visualization
#
# Pada bagian ini, kita akan melakukan visualisasi data untuk mendapatkan wawasan awal tentang dataset.

# %% [markdown]
# ### 3.3.1 Distribusi Pemesanan berdasarkan Jenis Hotel

# %%
# Distribusi pemesanan berdasarkan jenis hotel
plt.figure(figsize=(10, 6))
sns.countplot(x='hotel', data=df, palette='viridis')
plt.title('Distribusi Pemesanan berdasarkan Jenis Hotel', fontsize=16)
plt.xlabel('Jenis Hotel', fontsize=12)
plt.ylabel('Jumlah Pemesanan', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

# %% [markdown]
# ### 3.3.2 Distribusi Pemesanan yang Dibatalkan vs Tidak Dibatalkan

# %%
# Distribusi pemesanan yang dibatalkan vs tidak dibatalkan
plt.figure(figsize=(10, 6))
sns.countplot(x='is_canceled', data=df, palette='viridis')
plt.title('Distribusi Pemesanan yang Dibatalkan vs Tidak Dibatalkan', fontsize=16)
plt.xlabel('Status Pembatalan (0: Tidak Dibatalkan, 1: Dibatalkan)', fontsize=12)
plt.ylabel('Jumlah Pemesanan', fontsize=12)
plt.xticks([0, 1], ['Tidak Dibatalkan', 'Dibatalkan'], fontsize=10)
plt.yticks(fontsize=10)
plt.show()

# %%
# Persentase pemesanan yang dibatalkan vs tidak dibatalkan
canceled_percentage = df['is_canceled'].value_counts(normalize=True) * 100
plt.figure(figsize=(10, 6))
plt.pie(canceled_percentage, labels=['Tidak Dibatalkan', 'Dibatalkan'], autopct='%1.1f%%',
        colors=['#66b3ff', '#ff9999'], startangle=90, explode=(0.1, 0))
plt.title('Persentase Pemesanan yang Dibatalkan vs Tidak Dibatalkan', fontsize=16)
plt.axis('equal')
plt.show()

# %% [markdown]
# ### 3.3.3 Distribusi Pemesanan berdasarkan Bulan Kedatangan

# %%
# Distribusi pemesanan berdasarkan bulan kedatangan
plt.figure(figsize=(14, 8))
sns.countplot(x='arrival_date_month', data=df, palette='viridis',
              order=['January', 'February', 'March', 'April', 'May', 'June',
                     'July', 'August', 'September', 'October', 'November', 'December'])
plt.title('Distribusi Pemesanan berdasarkan Bulan Kedatangan', fontsize=16)
plt.xlabel('Bulan Kedatangan', fontsize=12)
plt.ylabel('Jumlah Pemesanan', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.show()

# %% [markdown]
# ### 3.3.4 Distribusi Pemesanan berdasarkan Jenis Hotel dan Status Pembatalan

# %%
# Distribusi pemesanan berdasarkan jenis hotel dan status pembatalan
plt.figure(figsize=(12, 6))
sns.countplot(x='hotel', hue='is_canceled', data=df, palette='viridis')
plt.title('Distribusi Pemesanan berdasarkan Jenis Hotel dan Status Pembatalan', fontsize=16)
plt.xlabel('Jenis Hotel', fontsize=12)
plt.ylabel('Jumlah Pemesanan', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(title='Status Pembatalan', labels=['Tidak Dibatalkan', 'Dibatalkan'])
plt.show()

# %% [markdown]
# ### 3.3.5 Distribusi Pemesanan berdasarkan Segmen Pasar

# %%
# Distribusi pemesanan berdasarkan segmen pasar
plt.figure(figsize=(14, 8))
sns.countplot(x='market_segment', data=df, palette='viridis')
plt.title('Distribusi Pemesanan berdasarkan Segmen Pasar', fontsize=16)
plt.xlabel('Segmen Pasar', fontsize=12)
plt.ylabel('Jumlah Pemesanan', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.show()

# %% [markdown]
# ### 3.3.6 Distribusi Pemesanan berdasarkan Negara Asal Tamu

# %%
# Distribusi pemesanan berdasarkan negara asal tamu (top 15)
top_countries = df['country'].value_counts().head(15)
plt.figure(figsize=(14, 8))
sns.barplot(x=top_countries.index, y=top_countries.values, palette='viridis')
plt.title('Distribusi Pemesanan berdasarkan Negara Asal Tamu (Top 15)', fontsize=16)
plt.xlabel('Negara', fontsize=12)
plt.ylabel('Jumlah Pemesanan', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.show()

# %% [markdown]
# ### 3.3.7 Distribusi Tarif Harian Rata-rata (ADR)

# %%
# Distribusi tarif harian rata-rata (ADR)
plt.figure(figsize=(14, 8))
sns.histplot(df['adr'], kde=True, bins=50, palette='viridis')
plt.title('Distribusi Tarif Harian Rata-rata (ADR)', fontsize=16)
plt.xlabel('ADR', fontsize=12)
plt.ylabel('Frekuensi', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

# %% [markdown]
# ### 3.3.8 Distribusi Lead Time

# %%
# Distribusi lead time
plt.figure(figsize=(14, 8))
sns.histplot(df['lead_time'], kde=True, bins=50, palette='viridis')
plt.title('Distribusi Lead Time', fontsize=16)
plt.xlabel('Lead Time (hari)', fontsize=12)
plt.ylabel('Frekuensi', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

# %% [markdown]
# ### 3.3.9 Korelasi antar Fitur Numerik

# %%
# Korelasi antar fitur numerik
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
correlation_matrix = df[numeric_columns].corr()

plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', linewidths=0.5, fmt='.2f')
plt.title('Korelasi antar Fitur Numerik', fontsize=16)
plt.xticks(fontsize=10, rotation=90)
plt.yticks(fontsize=10)
plt.show()

# %% [markdown]
# ### 3.3.10 Analisis Tarif Harian Rata-rata (ADR) berdasarkan Jenis Hotel dan Bulan Kedatangan

# %%
# Analisis tarif harian rata-rata (ADR) berdasarkan jenis hotel dan bulan kedatangan
plt.figure(figsize=(16, 8))
sns.boxplot(x='arrival_date_month', y='adr', hue='hotel', data=df, palette='viridis',
            order=['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December'])
plt.title('Tarif Harian Rata-rata (ADR) berdasarkan Jenis Hotel dan Bulan Kedatangan', fontsize=16)
plt.xlabel('Bulan Kedatangan', fontsize=12)
plt.ylabel('ADR', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.legend(title='Jenis Hotel')
plt.show()

# %% [markdown]
# ### 3.3.11 Analisis Pembatalan berdasarkan Lead Time

# %%
# Analisis pembatalan berdasarkan lead time
plt.figure(figsize=(14, 8))
sns.boxplot(x='is_canceled', y='lead_time', data=df, palette='viridis')
plt.title('Pembatalan berdasarkan Lead Time', fontsize=16)
plt.xlabel('Status Pembatalan (0: Tidak Dibatalkan, 1: Dibatalkan)', fontsize=12)
plt.ylabel('Lead Time (hari)', fontsize=12)
plt.xticks([0, 1], ['Tidak Dibatalkan', 'Dibatalkan'], fontsize=10)
plt.yticks(fontsize=10)
plt.show()

# %% [markdown]
# ### 3.3.12 Analisis Lama Menginap

# %%
# Menambahkan kolom total_nights untuk analisis lama menginap
df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']

# Analisis lama menginap
plt.figure(figsize=(14, 8))
sns.histplot(df['total_nights'], kde=True, bins=30, palette='viridis')
plt.title('Distribusi Lama Menginap', fontsize=16)
plt.xlabel('Jumlah Malam', fontsize=12)
plt.ylabel('Frekuensi', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlim(0, 20)  # Membatasi tampilan untuk melihat pola yang lebih jelas
plt.show()

# %%
# Analisis lama menginap berdasarkan jenis hotel
plt.figure(figsize=(14, 8))
sns.boxplot(x='hotel', y='total_nights', data=df, palette='viridis')
plt.title('Lama Menginap berdasarkan Jenis Hotel', fontsize=16)
plt.xlabel('Jenis Hotel', fontsize=12)
plt.ylabel('Jumlah Malam', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.ylim(0, 15)  # Membatasi tampilan untuk melihat pola yang lebih jelas
plt.show()

# %% [markdown]
# ## Insight dari Eksplorasi Data
#
# Berdasarkan eksplorasi data yang telah dilakukan, beberapa insight yang dapat diperoleh adalah:
#
# 1. **Distribusi Jenis Hotel**: City Hotel memiliki jumlah pemesanan yang lebih banyak dibandingkan Resort Hotel.
#
# 2. **Tingkat Pembatalan**: Sekitar 37% dari total pemesanan dibatalkan, yang merupakan angka yang cukup signifikan dan dapat mempengaruhi pendapatan hotel.
#
# 3. **Pola Musiman**: Terdapat pola musiman dalam pemesanan hotel, dengan puncak pemesanan terjadi pada bulan Juli dan Agustus (musim panas di belahan bumi utara).
#
# 4. **Segmen Pasar**: Segmen pasar "Online TA" (Online Travel Agency) mendominasi pemesanan, menunjukkan pentingnya platform online dalam industri perhotelan.
#
# 5. **Negara Asal Tamu**: Portugal (negara tempat hotel berada) merupakan negara asal tamu terbanyak, diikuti oleh negara-negara Eropa lainnya.
#
# 6. **Tarif Harian Rata-rata (ADR)**: Distribusi ADR menunjukkan variasi yang cukup besar, dengan beberapa outlier yang memiliki nilai sangat tinggi.
#
# 7. **Lead Time**: Pemesanan yang dibatalkan cenderung memiliki lead time yang lebih panjang, menunjukkan bahwa pemesanan yang dilakukan jauh-jauh hari memiliki risiko pembatalan yang lebih tinggi.
#
# 8. **Lama Menginap**: Sebagian besar tamu menginap untuk jangka waktu yang singkat (1-4 malam), dengan pola yang berbeda antara City Hotel dan Resort Hotel.
#
# 9. **Korelasi antar Fitur**: Terdapat beberapa korelasi yang menarik antar fitur, seperti korelasi positif antara lead time dan pembatalan, serta korelasi negatif antara is_repeated_guest dan pembatalan.
#
# 10. **Variasi Harga Musiman**: Tarif hotel bervariasi berdasarkan bulan, dengan Resort Hotel menunjukkan variasi yang lebih besar dibandingkan City Hotel.
