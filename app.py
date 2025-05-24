import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import pandas as pd
from sklearn.decomposition import PCA

# Load model
model_path = os.path.join("models", "kmeans_model.pkl")
model = joblib.load(model_path)

st.title("Segmentasi Pelanggan dengan Clustering (KMeans)")

st.write(f"Model ini dilatih dengan {model.n_features_in_} fitur")
st.write("Masukkan data pelanggan:")

# Input fitur
avg_credit = st.slider("Batas/Limit Credit Rata-Rata", 3000.0, 34574.0, 15000.0)
total_credit_card = st.slider("Jumlah Kartu Kredit", 1.0, 10.0, 4.0)
total_visit_bank = st.slider("Jumlah Kunjungan ke Bank", 0.0, 5.0, 2.0)
total_visit_online = st.slider("Jumlah Kunjungan Online", 0.0, 15.0, 5.0)
total_calls_made = st.slider("Jumlah Panggilan", 0.0, 10.0, 3.0)

# Buat dataframe dengan nama kolom sesuai yang kamu mau
data = {
    'Avg_Credit_Limit': [avg_credit],
    'Total_Credit_Cards': [total_credit_card],
    'Total_visits_bank': [total_visit_bank],
    'Total_visits_online': [total_visit_online],
    'Total_calls_made': [total_calls_made]
}

df = pd.DataFrame(data)

st.write(df)  # Tampilkan dataframe

# Pastikan fitur yang dipakai model sama dengan ini
fitur = ['Avg_Credit_Limit', 'Total_Credit_Cards', 'Total_visits_bank', 'Total_visits_online', 'Total_calls_made']

# Prediksi cluster dengan model
pred_cluster = model.predict(df[fitur])

# Prediksi klaster
# pred_cluster = model.predict(df[['avg_credit', 'total_credit_card', 'total_visit_bank', 'total_visit_online', 'total_calls_made']])[0]
st.success(f"Pelanggan ini termasuk dalam **Cluster {pred_cluster}**")

# st.write(f"## Data Pelanggan dalam ClusterM {pred_cluster}")
# # ---- Visualisasi Cluster ----
# st.subheader("Visualisasi Klaster Pelanggan (PCA 2D)")

# # Dummy data (untuk simulasi visualisasi klaster)
# np.random.seed(42)
# dummy_data = np.vstack([
#     np.random.normal([10000, 3, 1, 2, 1], [2000, 1, 1, 1, 1], size=(50, 5)),
#     np.random.normal([20000, 5, 2, 4, 2], [2000, 1, 1, 1, 1], size=(50, 5)),
#     np.random.normal([30000, 7, 3, 6, 3], [2000, 1, 1, 1, 1], size=(50, 5)),
# ])

# dummy_labels = model.predict(dummy_data)

# # Gabungkan data dummy dan input user
# input_array = input_df.to_numpy()
# full_data = np.vstack([dummy_data, input_array])
# full_labels = np.append(dummy_labels, pred_cluster)

# # PCA untuk reduksi dimensi ke 2D
# pca = PCA(n_components=2)
# reduced_data = pca.fit_transform(full_data)

# # Plot klaster
# fig, ax = plt.subplots()
# colors = ['green', 'blue', 'orange', 'purple', 'brown']

# for i in range(model.n_clusters):
#     points = reduced_data[full_labels == i]
#     ax.scatter(points[:, 0], points[:, 1], color=colors[i], label=f"Cluster {i}", alpha=0.5)

# # Tandai input user
# ax.scatter(reduced_data[-1, 0], reduced_data[-1, 1], c='black', s=150, edgecolors='white', label="Input Anda")

# ax.set_xlabel("Komponen Utama 1")
# ax.set_ylabel("Komponen Utama 2")
# ax.legend()
# st.pyplot(fig)
