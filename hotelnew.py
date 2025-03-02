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

# %% [markdown]
# # 4. Data Preparation
#
# Pada bagian ini, kita akan melakukan persiapan data untuk analisis lebih lanjut atau pelatihan model machine learning. Langkah-langkah yang akan dilakukan meliputi:
#
# 1. Handling Missing Values
# 2. Handling Outliers
# 3. Encoding Categorical Variables
# 4. Feature Engineering
# 5. Feature Selection

# %% [markdown]
# ## 4.1 Handling Missing Values
#
# Pertama, kita akan menangani nilai yang hilang (missing values) dalam dataset.

# %%
# Melihat kembali jumlah nilai yang hilang untuk setiap kolom
missing_values = pd.DataFrame({
    'Jumlah Nilai Hilang': df.isna().sum(),
    'Persentase Nilai Hilang': (df.isna().sum() / len(df) * 100).round(2)
})

missing_values = missing_values.sort_values('Jumlah Nilai Hilang', ascending=False)
missing_values[missing_values['Jumlah Nilai Hilang'] > 0]

# %% [markdown]
# Berdasarkan analisis nilai yang hilang, kita perlu menangani beberapa kolom dengan nilai yang hilang:
#
# 1. **children**: Kolom ini memiliki beberapa nilai yang hilang. Kita akan mengisi nilai yang hilang dengan 0, dengan asumsi bahwa jika tidak ada informasi tentang jumlah anak, maka tidak ada anak dalam pemesanan.
#
# 2. **country**: Kolom ini memiliki beberapa nilai yang hilang. Kita akan mengisi nilai yang hilang dengan "Unknown", menunjukkan bahwa negara asal tamu tidak diketahui.
#
# 3. **agent**: Kolom ini memiliki banyak nilai yang hilang. Kita akan mengisi nilai yang hilang dengan 0, menunjukkan bahwa pemesanan tidak dilakukan melalui agen.
#
# 4. **company**: Kolom ini memiliki banyak nilai yang hilang. Kita akan mengisi nilai yang hilang dengan 0, menunjukkan bahwa pemesanan tidak dilakukan melalui perusahaan.

# %%
# Membuat salinan dataset untuk persiapan data
df_prep = df.copy()

# Mengisi nilai yang hilang
df_prep['children'] = df_prep['children'].fillna(0)
df_prep['country'] = df_prep['country'].fillna('Unknown')
df_prep['agent'] = df_prep['agent'].fillna(0)
df_prep['company'] = df_prep['company'].fillna(0)

# Memeriksa apakah masih ada nilai yang hilang
missing_values_after = pd.DataFrame({
    'Jumlah Nilai Hilang': df_prep.isna().sum(),
    'Persentase Nilai Hilang': (df_prep.isna().sum() / len(df_prep) * 100).round(2)
})

missing_values_after = missing_values_after.sort_values('Jumlah Nilai Hilang', ascending=False)
missing_values_after[missing_values_after['Jumlah Nilai Hilang'] > 0]

# %% [markdown]
# Selain menangani nilai yang hilang, kita juga perlu menangani kasus di mana jumlah tamu (adults + children + babies) adalah 0, yang merupakan anomali dalam data.

# %%
# Menghitung jumlah baris di mana jumlah tamu adalah 0
zero_guests = df_prep[(df_prep['adults'] + df_prep['children'] + df_prep['babies']) == 0].shape[0]
print(f"Jumlah baris di mana jumlah tamu adalah 0: {zero_guests}")

# Menghapus baris di mana jumlah tamu adalah 0
df_prep = df_prep[(df_prep['adults'] + df_prep['children'] + df_prep['babies']) > 0]
print(f"Jumlah baris setelah menghapus baris dengan jumlah tamu 0: {df_prep.shape[0]}")

# %% [markdown]
# ## 4.2 Handling Outliers
#
# Selanjutnya, kita akan menangani outlier dalam dataset, terutama pada kolom 'adr' (Average Daily Rate) yang memiliki beberapa nilai ekstrem.

# %%
# Melihat statistik dasar untuk kolom 'adr'
df_prep['adr'].describe()

# %%
# Visualisasi distribusi 'adr' dengan boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(y='adr', data=df_prep)
plt.title('Boxplot untuk ADR', fontsize=16)
plt.ylabel('ADR', fontsize=12)
plt.yticks(fontsize=10)
plt.show()

# %% [markdown]
# Dari visualisasi di atas, terlihat bahwa kolom 'adr' memiliki beberapa outlier dengan nilai yang sangat tinggi. Kita akan menangani outlier ini dengan metode capping, yaitu membatasi nilai maksimum pada persentil ke-99.

# %%
# Menangani outlier pada kolom 'adr' dengan metode capping
adr_cap = np.percentile(df_prep['adr'], 99)
print(f"Nilai capping untuk ADR: {adr_cap}")

df_prep['adr'] = np.where(df_prep['adr'] > adr_cap, adr_cap, df_prep['adr'])

# Melihat statistik dasar untuk kolom 'adr' setelah capping
df_prep['adr'].describe()

# %%
# Visualisasi distribusi 'adr' setelah capping
plt.figure(figsize=(10, 6))
sns.boxplot(y='adr', data=df_prep)
plt.title('Boxplot untuk ADR setelah Capping', fontsize=16)
plt.ylabel('ADR', fontsize=12)
plt.yticks(fontsize=10)
plt.show()

# %% [markdown]
# ## 4.3 Encoding Categorical Variables
#
# Selanjutnya, kita akan melakukan encoding terhadap variabel kategorikal agar dapat digunakan dalam model machine learning.

# %%
# Melihat kolom-kolom kategorikal
categorical_columns = df_prep.select_dtypes(include=['object']).columns
print(f"Kolom-kolom kategorikal: {list(categorical_columns)}")

# %% [markdown]
# Kita akan melakukan encoding terhadap variabel kategorikal dengan metode yang sesuai:
#
# 1. **One-Hot Encoding**: Untuk variabel kategorikal dengan kardinalitas rendah (jumlah kategori sedikit), seperti 'hotel', 'meal', 'market_segment', 'distribution_channel', 'reserved_room_type', 'assigned_room_type', 'deposit_type', 'customer_type', dan 'reservation_status'.
#
# 2. **Label Encoding**: Untuk variabel kategorikal dengan kardinalitas tinggi (jumlah kategori banyak), seperti 'country'.
#
# 3. **Ordinal Encoding**: Untuk variabel kategorikal yang memiliki urutan, seperti 'arrival_date_month'.

# %%
# One-Hot Encoding untuk variabel kategorikal dengan kardinalitas rendah
categorical_columns_low_cardinality = ['hotel', 'meal', 'market_segment', 'distribution_channel',
                                       'reserved_room_type', 'assigned_room_type', 'deposit_type',
                                       'customer_type', 'reservation_status']

df_encoded = pd.get_dummies(df_prep, columns=categorical_columns_low_cardinality, drop_first=True)

# %%
# Label Encoding untuk variabel kategorikal dengan kardinalitas tinggi
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df_encoded['country_encoded'] = label_encoder.fit_transform(df_prep['country'])
df_encoded.drop('country', axis=1, inplace=True)

# %%
# Ordinal Encoding untuk variabel kategorikal yang memiliki urutan
month_order = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
               'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}

df_encoded['arrival_date_month_encoded'] = df_prep['arrival_date_month'].map(month_order)
df_encoded.drop('arrival_date_month', axis=1, inplace=True)

# %%
# Melihat hasil encoding
df_encoded.head()

# %% [markdown]
# ## 4.4 Feature Engineering
#
# Selanjutnya, kita akan melakukan feature engineering untuk menciptakan fitur baru yang mungkin berguna untuk analisis atau model machine learning.

# %%
# 1. Total Nights: Jumlah total malam menginap
df_encoded['total_nights'] = df_encoded['stays_in_weekend_nights'] + df_encoded['stays_in_week_nights']

# 2. Total Guests: Jumlah total tamu
df_encoded['total_guests'] = df_encoded['adults'] + df_encoded['children'] + df_encoded['babies']

# 3. Has Children: Apakah pemesanan termasuk anak-anak
df_encoded['has_children'] = (df_encoded['children'] > 0).astype(int)

# 4. Has Babies: Apakah pemesanan termasuk bayi
df_encoded['has_babies'] = (df_encoded['babies'] > 0).astype(int)

# 5. ADR per Person: Tarif harian rata-rata per orang
df_encoded['adr_per_person'] = df_encoded['adr'] / df_encoded['total_guests']
df_encoded['adr_per_person'] = df_encoded['adr_per_person'].replace([np.inf, -np.inf], np.nan).fillna(df_encoded['adr'])

# 6. Is Weekend Stay: Apakah pemesanan termasuk menginap di akhir pekan
df_encoded['is_weekend_stay'] = (df_encoded['stays_in_weekend_nights'] > 0).astype(int)

# 7. Is Long Stay: Apakah pemesanan termasuk menginap dalam jangka waktu yang panjang (> 7 malam)
df_encoded['is_long_stay'] = (df_encoded['total_nights'] > 7).astype(int)

# 8. Arrival Quarter: Kuartal kedatangan
df_encoded['arrival_quarter'] = ((df_encoded['arrival_date_month_encoded'] - 1) // 3 + 1).astype(int)

# 9. Is High Season: Apakah pemesanan pada musim tinggi (Juni-Agustus)
df_encoded['is_high_season'] = ((df_encoded['arrival_date_month_encoded'] >= 6) &
                               (df_encoded['arrival_date_month_encoded'] <= 8)).astype(int)

# 10. Lead Time Category: Kategori lead time
df_encoded['lead_time_category'] = pd.cut(df_encoded['lead_time'],
                                         bins=[0, 7, 30, 90, 365, float('inf')],
                                         labels=[1, 2, 3, 4, 5])

# %%
# Melihat hasil feature engineering
df_encoded.head()

# %% [markdown]
# ## 4.5 Feature Selection
#
# Terakhir, kita akan melakukan feature selection untuk memilih fitur yang paling relevan untuk model machine learning.

# %%
# Melihat korelasi fitur dengan target (is_canceled)
correlation_with_target = df_encoded.corr()['is_canceled'].sort_values(ascending=False)
print("Korelasi Fitur dengan Target (is_canceled):")
print(correlation_with_target)

# %%
# Visualisasi korelasi fitur dengan target
plt.figure(figsize=(14, 10))
correlation_with_target.drop('is_canceled').sort_values().plot(kind='barh')
plt.title('Korelasi Fitur dengan Target (is_canceled)', fontsize=16)
plt.xlabel('Korelasi', fontsize=12)
plt.ylabel('Fitur', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True)
plt.show()

# %% [markdown]
# Berdasarkan analisis korelasi, kita dapat memilih fitur-fitur yang memiliki korelasi yang signifikan dengan target (is_canceled). Kita akan memilih fitur-fitur dengan nilai absolut korelasi di atas threshold tertentu, misalnya 0.05.

# %%
# Memilih fitur berdasarkan korelasi
threshold = 0.05
selected_features = correlation_with_target.drop('is_canceled').abs()[correlation_with_target.drop('is_canceled').abs() > threshold].index.tolist()
print(f"Jumlah fitur yang dipilih: {len(selected_features)}")
print(f"Fitur yang dipilih: {selected_features}")

# %%
# Membuat dataset final dengan fitur yang dipilih
X = df_encoded[selected_features]
y = df_encoded['is_canceled']

print(f"Dimensi X: {X.shape}")
print(f"Dimensi y: {y.shape}")

# %% [markdown]
# # 5. Conclusion
#
# Dalam analisis ini, kita telah melakukan persiapan data yang komprehensif untuk dataset Hotel Booking Demand. Berikut adalah ringkasan langkah-langkah yang telah dilakukan:
#
# 1. **Data Description**: Kita telah menjelaskan dataset, termasuk nama, sumber, deskripsi singkat, jumlah data, dan format data.
#
# 2. **Data Loading**: Kita telah memuat dataset ke dalam lingkungan pemrograman Python menggunakan Pandas dan menjelaskan tantangan yang mungkin dihadapi saat memuat data.
#
# 3. **Data Understanding**: Kita telah melakukan eksplorasi data untuk memahami karakteristik dan pola dalam dataset, termasuk statistik dasar, analisis nilai yang hilang, dan visualisasi data.
#
# 4. **Data Preparation**: Kita telah melakukan persiapan data untuk analisis lebih lanjut atau pelatihan model machine learning, termasuk:
#    - Handling Missing Values: Mengisi nilai yang hilang dengan nilai yang sesuai.
#    - Handling Outliers: Menangani outlier pada kolom 'adr' dengan metode capping.
#    - Encoding Categorical Variables: Melakukan encoding terhadap variabel kategorikal dengan metode yang sesuai.
#    - Feature Engineering: Menciptakan fitur baru yang mungkin berguna untuk analisis atau model machine learning.
#    - Feature Selection: Memilih fitur yang paling relevan untuk model machine learning.
#
# Dataset Hotel Booking Demand ini sangat kaya dengan informasi dan dapat digunakan untuk berbagai analisis dan model machine learning, seperti:
#
# - Prediksi pembatalan pemesanan hotel
# - Analisis faktor-faktor yang mempengaruhi pembatalan
# - Segmentasi pelanggan
# - Analisis pola musiman dalam pemesanan hotel
# - Optimasi harga dan strategi pemasaran
#
# Dengan persiapan data yang telah dilakukan, dataset ini siap untuk digunakan dalam analisis lebih lanjut atau pelatihan model machine learning.

# %% [markdown]
# ## Keputusan yang Diambil dalam Preprocessing dan Alasannya
#
# 1. **Handling Missing Values**:
#    - Mengisi nilai yang hilang pada kolom 'children' dengan 0, dengan asumsi bahwa jika tidak ada informasi tentang jumlah anak, maka tidak ada anak dalam pemesanan.
#    - Mengisi nilai yang hilang pada kolom 'country' dengan "Unknown", menunjukkan bahwa negara asal tamu tidak diketahui.
#    - Mengisi nilai yang hilang pada kolom 'agent' dan 'company' dengan 0, menunjukkan bahwa pemesanan tidak dilakukan melalui agen atau perusahaan.
#    - Alasan: Pendekatan ini memungkinkan kita untuk mempertahankan semua baris data tanpa menghapus informasi yang berharga.
#
# 2. **Handling Outliers**:
#    - Menangani outlier pada kolom 'adr' dengan metode capping, yaitu membatasi nilai maksimum pada persentil ke-99.
#    - Alasan: Metode ini memungkinkan kita untuk menangani nilai ekstrem tanpa menghapus terlalu banyak data, sehingga mempertahankan informasi yang berharga.
#
# 3. **Encoding Categorical Variables**:
#    - Menggunakan One-Hot Encoding untuk variabel kategorikal dengan kardinalitas rendah.
#    - Menggunakan Label Encoding untuk variabel kategorikal dengan kardinalitas tinggi.
#    - Menggunakan Ordinal Encoding untuk variabel kategorikal yang memiliki urutan.
#    - Alasan: Pendekatan ini memungkinkan kita untuk mengubah variabel kategorikal menjadi format numerik yang dapat digunakan dalam model machine learning, dengan mempertimbangkan karakteristik masing-masing variabel.
#
# 4. **Feature Engineering**:
#    - Menciptakan fitur baru seperti 'total_nights', 'total_guests', 'has_children', 'has_babies', 'adr_per_person', 'is_weekend_stay', 'is_long_stay', 'arrival_quarter', 'is_high_season', dan 'lead_time_category'.
#    - Alasan: Fitur-fitur baru ini dapat memberikan informasi tambahan yang mungkin berguna untuk analisis atau model machine learning, seperti pola musiman, karakteristik pemesanan, dan perilaku pelanggan.
#
# 5. **Feature Selection**:
#    - Memilih fitur berdasarkan korelasi dengan target (is_canceled), dengan threshold 0.05.
#    - Alasan: Pendekatan ini memungkinkan kita untuk fokus pada fitur-fitur yang memiliki hubungan yang signifikan dengan target, sehingga dapat meningkatkan performa model dan mengurangi dimensi data.
#
# Semua keputusan di atas diambil dengan mempertimbangkan karakteristik dataset, tujuan analisis, dan praktik terbaik dalam persiapan data untuk analisis atau model machine learning.