import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.set_page_config(page_title="Profilo Cliente PetCare", layout="centered")

# ================================
# 1. DATI FISSI PER OGNI CLUSTER
# ================================
np.random.seed(42)

cluster_0 = np.random.normal([100, 10], [3, 1], size=(50, 2))  # Cacciatori di offerte
cluster_1 = np.random.normal([115, 13], [3, 1], size=(50, 2))  # Pet lover organizzati
cluster_2 = np.random.normal([130, 5],  [3, 1], size=(50, 2))  # Animali affezionati
cluster_3 = np.random.normal([160, 4],  [3, 1], size=(50, 2))  # Amici animali premium

X = np.vstack((cluster_0, cluster_1, cluster_2, cluster_3))

# ================================
# 2. CLUSTERING
# ================================
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X)
centroidi = kmeans.cluster_centers_

# ================================
# 3. MAPPING DEI NOMI AI CLUSTER
# ================================
etichette = [
    "Cacciatori di offerte",
    "Pet lover organizzati",
    "Animali affezionati",
    "Amici animali premium"
]

# Centroidi veri usati per generare i cluster
centroidi_reali = np.array([
    [100, 10],
    [115, 13],
    [130, 5],
    [160, 4]
])

# Calcola distanza da centroidi del modello ai centroidi reali
distanze = np.linalg.norm(centroidi[:, None] - centroidi_reali[None, :], axis=2)
mappatura = {}
for i, indice in enumerate(np.argmin(distanze, axis=0)):
    mappatura[indice] = etichette[i]

# ================================
# 4. STREAMLIT UI
# ================================
st.title("🐾 Scopri il tuo profilo cliente PetCare!")

st.markdown("Inserisci la **spesa media mensile** e il **numero di acquisti** per scoprire a quale cluster appartieni e quali strategie sono più adatte a te.")

spesa = st.number_input("💰 Spesa media mensile (€)", min_value=0.0, step=1.0)
acquisti = st.number_input("🛍️ Numero di acquisti mensili", min_value=0, step=1)

if spesa and acquisti:
    nuovo_cliente = np.array([[spesa, acquisti]])
    cluster_assegnato = kmeans.predict(nuovo_cliente)[0]
    nome_cluster = mappatura[cluster_assegnato]

    st.markdown(f"🎯 **Il tuo cluster è:** 🐶 **{nome_cluster}**")

    # Strategie esempio (puoi personalizzarle)
    strategie = {
        "Cacciatori di offerte": [
            "📩 Newsletter settimanale",
            "🎁 Bundle 2+1 snack",
            "🏆 Programma punti",
            "🛒 Offerte in homepage"
        ],
        "Pet lover organizzati": [
            "📅 Abbonamenti personalizzati",
            "🔔 Reminder per riacquisti",
            "🆕 Early access novità"
        ],
        "Animali affezionati": [
            "💌 Email mensile personalizzata",
            "🎉 Sconti ricorrenze",
            "💎 Up-selling mirato",
            "🧪 Test nuovi prodotti"
        ],
        "Amici animali premium": [
            "💎 Upselling mirato",
            "🎁 Packaging premium",
            "📚 Storytelling via email",
            "📛 Programma VIP"
        ]
    }

    st.markdown("📌 **Strategie consigliate per questo profilo:**")
    for s in strategie[nome_cluster]:
        st.markdown(f"- {s}")

    # ================================
    # 5. PLOT CON NUOVO CLIENTE
    # ================================
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='Pastel1', alpha=0.6)
    plt.scatter(centroidi[:, 0], centroidi[:, 1], c='red', s=200, marker='X', label='Centroidi')
    plt.scatter(spesa, acquisti, c='black', marker='*', s=200, label='Nuovo Cliente')

    plt.xlabel("Spesa media mensile (€)")
    plt.ylabel("Numero di acquisti mensili")
    plt.title("Segmentazione clienti con K-Means")
    plt.legend()
    st.pyplot(plt)
