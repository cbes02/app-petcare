import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Dati simulati: X = [spesa media, numero acquisti]
np.random.seed(42)
cluster1 = np.random.normal([100, 10], [5, 2], size=(50, 2))
cluster2 = np.random.normal([115, 13], [5, 2], size=(50, 2))
cluster3 = np.random.normal([130, 5], [5, 2], size=(50, 2))
cluster4 = np.random.normal([160, 4], [5, 2], size=(50, 2))

X = np.vstack((cluster1, cluster2, cluster3, cluster4))

# Applica il modello KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)

# Descrizioni dei cluster
cluster_descrizioni = {
    0: ("💌 Cacciatori di offerte", [
        "📰 Newsletter settimanale",
        "🎁 Bundle 2+1 snack",
        "🏆 Programma punti",
        "🛒 Offerte in homepage"
    ]),
    1: ("📅 Per lover organizzati", [
        "📦 Abbonamenti personalizzati",
        "📧 Reminder riacquisto",
        "🚀 Promo early-access"
    ]),
    2: ("🐾 Animali affezionati", [
        "💌 Email mensile personalizzata",
        "🎉 Sconti ricorrenze",
        "💎 Up-selling mirato",
        "🧪 Test nuovi prodotti"
    ]),
    3: ("👑 Amici animali premium", [
        "💎 Upselling mirato",
        "🎁 Packaging premium",
        "📚 Storytelling via email",
        "💎 Programma VIP"
    ])
}

# UI
st.title("🐾 Scopri il tuo profilo cliente PetCare!")
st.write("Inserisci la **spesa media mensile** e il **numero di acquisti** per scoprire a quale cluster appartieni e quali strategie sono più adatte a te.")

spesa = st.number_input("💰 Spesa media mensile (€)", min_value=0.0, step=10.0)
acquisti = st.number_input("🛍️ Numero di acquisti mensili", min_value=0, step=1)

if spesa > 0 and acquisti > 0:
    nuovo_cliente = np.array([[spesa, acquisti]])
    cluster = kmeans.predict(nuovo_cliente)[0]

    nome_cluster, strategie = cluster_descrizioni[cluster]

    st.markdown(f"🎯 **Il tuo cluster è:** {nome_cluster}")
    st.markdown("📌 **Strategie consigliate per questo profilo:**")
    for s in strategie:
        st.write(f"- {s}")

    # Grafico
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='tab10', alpha=0.6)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, marker='X', label='Centroidi')
    plt.scatter(spesa, acquisti, color='black', s=100, marker='*', label='Nuovo Cliente')
    plt.xlabel("Spesa media mensile")
    plt.ylabel("Numero di acquisti mensili")
    plt.title("Segmentazione clienti con K-Means")
    plt.legend()
    st.pyplot(plt)
