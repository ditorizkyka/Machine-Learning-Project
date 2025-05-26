import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import pandas as pd
from sklearn.decomposition import PCA

@st.cache_resource
def load_models():
    kmeans_model = joblib.load('models/kmeans_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return kmeans_model, scaler

try:
    kmeans_model, scaler = load_models()
    st.title("Segmentasi Pelanggan dengan Clustering (KMeans)")
    st.write(f"Model ini dilatih dengan {kmeans_model.n_features_in_} fitur dan {kmeans_model.n_clusters} cluster")
    
    st.write("## Masukkan data pelanggan:")
    
    col1, col2 = st.columns(2)

    with col1:
        avg_credit = st.number_input(
            "Batas/Limit Credit Rata-Rata",
            min_value=3000.0,
            max_value=200000.0,
            value=18000.0,
            step=1000.0
        )
        total_credit_card = st.slider(
            "Jumlah Kartu Kredit",
            min_value=1,
            max_value=10,
            value=5
        )
        total_visit_bank = st.slider(
            "Jumlah Kunjungan ke Bank",
            min_value=0,
            max_value=5,
            value=2
        )

    with col2:
        total_visit_online = st.slider(
            "Jumlah Kunjungan Online",
            min_value=0,
            max_value=15,
            value=2
        )
        total_calls_made = st.slider(
            "Jumlah Panggilan",
            min_value=0,
            max_value=10,
            value=3
        )

    
    new_data = pd.DataFrame({
        'Avg_Credit_Limit': [avg_credit],
        'Total_Credit_Cards': [total_credit_card],
        'Total_visits_bank': [total_visit_bank],
        'Total_visits_online': [total_visit_online],
        'Total_calls_made': [total_calls_made]
    })
    
    columns_to_scale = ['Avg_Credit_Limit', 'Total_Credit_Cards', 'Total_visits_bank', 
                       'Total_visits_online', 'Total_calls_made']
    
    if st.button("Prediksi Cluster", type="primary"):
        try:
            st.write("### Data yang Dimasukkan:")
            st.dataframe(new_data, use_container_width=True)
            
            new_data_scaled = scaler.transform(new_data[columns_to_scale])
            new_data_scaled_df = pd.DataFrame(new_data_scaled, columns=columns_to_scale)
            predicted_cluster = kmeans_model.predict(new_data_scaled_df)
            
            st.success(f"🎯 Pelanggan ini termasuk dalam **Cluster {predicted_cluster[0]}**")
            
            cluster_interpretation = {
                0: "High Credit Limit Group",
                1: "Medium Credit Limit Group", 
                2: "Low Value Customer"
            }
            
            if predicted_cluster[0] in cluster_interpretation:
                st.info(f"📊 **Interpretasi**: {cluster_interpretation[predicted_cluster[0]]}")
            
            st.write("### Visualisasi Posisi Pelanggan dalam Cluster")
            
            data_file_paths = ["customer_data.csv", "models/customer_data.csv", "data/Credit Card Customer Data.csv"]
            full_data = None
            
            for path in data_file_paths:
                if os.path.exists(path):
                    try:
                        full_data = pd.read_csv(path)
                        st.info(f"Dataset ditemukan: {path}")
                        break
                    except Exception as e:
                        continue
            
            if full_data is not None:
                required_columns = columns_to_scale
                if all(col in full_data.columns for col in required_columns):
                    if len(full_data) > 1000:
                        sample_data = full_data.sample(n=1000, random_state=42)
                    else:
                        sample_data = full_data
                    
                    try:
                        # Scale data lengkap
                        full_data_scaled = scaler.transform(sample_data[required_columns])
                        
                        # Prediksi cluster untuk data lengkap
                        full_labels = kmeans_model.predict(full_data_scaled)
                        
                        # Gabungkan data user dengan dataset untuk visualisasi
                        combined_data_scaled = np.vstack([full_data_scaled, new_data_scaled])
                        
                        # PCA untuk visualisasi 2D
                        pca = PCA(n_components=2)
                        reduced_data = pca.fit_transform(combined_data_scaled)
                        
                        # Plot
                        fig, ax = plt.subplots(figsize=(10, 8))
                        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
                        
                        # Plot semua cluster
                        for i in range(kmeans_model.n_clusters):
                            cluster_mask = full_labels == i
                            if np.any(cluster_mask):
                                cluster_points = reduced_data[:-1][cluster_mask]  # Exclude user point
                                ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                                         label=f"Cluster {i}", alpha=0.6, 
                                         color=colors[i % len(colors)], s=50)
                        
                        # Plot titik input user
                        user_point = reduced_data[-1]
                        ax.scatter(user_point[0], user_point[1], 
                                 color='red', s=300, edgecolors='black', 
                                 label="Input Anda", marker='*', linewidth=2)
                        
                        # Plot centroids
                        centroids_scaled = kmeans_model.cluster_centers_
                        centroids_pca = pca.transform(centroids_scaled)
                        ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                                 color='black', s=200, marker='x', 
                                 label='Centroids', linewidth=3)
                        
                        ax.set_xlabel(f"PCA Komponen 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
                        ax.set_ylabel(f"PCA Komponen 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
                        ax.set_title("Visualisasi Klaster Pelanggan")
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        st.pyplot(fig)
                        
                        # Informasi tambahan
                        st.write("### Informasi PCA")
                        st.write(f"- Komponen 1 menjelaskan {pca.explained_variance_ratio_[0]:.1%} dari variasi data")
                        st.write(f"- Komponen 2 menjelaskan {pca.explained_variance_ratio_[1]:.1%} dari variasi data")
                        st.write(f"- Total variasi yang dijelaskan: {sum(pca.explained_variance_ratio_):.1%}")
                        
                    except Exception as e:
                        st.error(f"Error dalam visualisasi: {str(e)}")
                        st.write("Menggunakan visualisasi sederhana...")
                        
                        # Visualisasi sederhana tanpa data lengkap
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.scatter(0, 0, color='red', s=300, marker='*', 
                                 label=f'Input Anda (Cluster {predicted_cluster[0]})')
                        ax.set_title("Prediksi Cluster Anda")
                        ax.legend()
                        st.pyplot(fig)
                else:
                    st.warning("Dataset tidak memiliki kolom yang diperlukan untuk visualisasi")
            else:
                st.warning("Dataset tidak ditemukan. Menampilkan hasil prediksi tanpa visualisasi.")
                
                # Tampilkan informasi cluster sederhana
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.text(0.5, 0.5, f'Cluster {predicted_cluster[0]}', 
                       ha='center', va='center', fontsize=24, fontweight='bold')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                ax.set_title("Hasil Prediksi")
                st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error dalam prediksi: {str(e)}")
            st.write("Pastikan file model dan scaler tersedia dan format input benar.")

except FileNotFoundError as e:
    st.error("❌ Model atau scaler tidak ditemukan!")
    st.write("Pastikan file berikut tersedia:")
    st.write("- `models/kmeans_model.pkl`")
    st.write("- `models/scaler.pkl`")
    st.write(f"Error detail: {str(e)}")
except Exception as e:
    st.error(f"Error dalam memuat aplikasi: {str(e)}")

# Sidebar dengan informasi tambahan
with st.sidebar:
    st.header("ℹ️ Informasi Aplikasi")
    st.write("""
    **Tentang Aplikasi:**
    - Menggunakan algoritma K-Means Clustering
    - Fitur input telah dinormalisasi
    - Visualisasi menggunakan PCA (Principal Component Analysis)
    
    **Cara Penggunaan:**
    1. Atur nilai untuk setiap fitur pelanggan
    2. Klik tombol 'Prediksi Cluster'
    3. Lihat hasilnya dan visualisasi
             
    Profil Pelanggan:
    - High Credit Limit Group: Segmen ini terdiri dari pelanggan bernilai tinggi yang merupakan digital-native users. Mereka memiliki kepercayaan tinggi dari bank (ditunjukkan dengan limit kredit besar) dan sudah terbiasa menggunakan layanan perbankan digital. Karakteristik mandiri dan tidak memerlukan banyak bantuan manual menunjukkan tingkat financial literacy yang baik.
    - Medium Credit Limit Group: Pelanggan dengan pendekatan tradisional yang masih mengandalkan layanan tatap muka di cabang bank. Meskipun memiliki nilai menengah, mereka menunjukkan loyalitas tinggi melalui kunjungan rutin ke bank. Segmen ini memiliki potensi besar untuk ditingkatkan nilainya melalui edukasi dan migrasi ke channel digital.
    - Low Credit Limit Group: Segmen dengan nilai transaksi rendah namun membutuhkan banyak dukungan customer service. Tingginya frekuensi panggilan dapat mengindikasikan kurangnya pemahaman terhadap produk/layanan atau ketidakpuasan. Dari perspektif cost-to-serve, segmen ini memerlukan pendekatan efisiensi operasional.
    """)
    
    st.header("🔧 Pengaturan")
    show_raw_data = st.checkbox("Tampilkan data mentah")
    
    if show_raw_data and 'new_data' in locals():
        st.write("**Data Input (Raw):**")
        st.json(new_data.to_dict('records')[0])