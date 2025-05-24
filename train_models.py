import numpy as np
from sklearn.cluster import KMeans
import pickle
import os

# Buat dummy data
np.random.seed(42)
data = np.vstack([
    np.random.normal([5, 10], [2, 5], size=(50, 2)),
    np.random.normal([15, 30], [2, 5], size=(50, 2)),
    np.random.normal([25, 80], [2, 5], size=(50, 2)),
])

model = KMeans(n_clusters=3, random_state=42)
model.fit(data)

# Simpan model
os.makedirs("models", exist_ok=True)
with open("models/kmeans_model.pkl", "wb") as f:
    pickle.dump(model, f)
