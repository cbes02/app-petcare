import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Titolo
st.title("🐾 Scopri il tuo profilo cliente PetCare!")

# Descrizione
st.write("Inserisci la **spesa media mensile** e il **numero di acquisti** per scoprire a quale cluster appartieni e quali strategie sono più adatte a te.")

# Input utente
spesa = st.number_input("💰 Spesa media mensile (€)", min_value=0.0, step=10.0, format="%.2f")
acquisti = st.number_input("🛍️ Numero di acquisti mensili", min_value=0, step=1)

# Dataset semplificato (dati finti simulati)
np.random.seed(42)
cluster1 = np.random.normal(loc=[100, 10], scale=[5, 2], size=(50, 2))   # Cacciatori di offerte
cluster2 = np.random.normal(loc=[115, 13], scale=[5, 2], size=(50, 2))   # Per lover organizzati
cluster3 = np.random.normal(loc=[130, 5], scale=[5, 2], size=(50, 2))    # Animali affezionati
cluster4 = np.random.normal(loc=[160, 4], scale=[5, 2], size=(50, 2))    # Amici animali premium

dati = np.vstack((cluster1, cluster2, cluster3, cluster4))

# Clustering
kmeans = KMeans(n_clusters=4, n_init=10)
kmeans.fit(dati)
centroidi = kmeans.cluster_centers_

# Classifica centroidi per posizione e assegna nomi corretti
centroidi_con_nomi = {
    "Cacciatori di offerte": None,
    "Per lover organizzati": None,
    "Animali affezionati": None,
    "Amici animali premium": None
}

for i, centroide in enumerate(centroidi):
    spesa_c, acquisti_c = centroide
    if spesa_c < 110 and acquisti_c >= 9:
        centroidi_con_nomi["Cacciatori di offerte"] = i
    elif spesa_c < 125 and acquisti_c > 11:
        centroidi_con_nomi["Per lover organizzati"] = i
    elif spesa_c > 125 and acquisti_c <= 6 and spesa_c < 150:
        centroidi_con_nomi["Animali affezionati"] = i
    elif spesa_c >= 150:
        centroidi_con_nomi["Amici animali premium"] = i

# Predizione nuovo cliente
nuovo_cliente = np.array([[spesa, acquisti]])
cluster_utente = kmeans.predict(nuovo_cliente)[0]

# Trova il nome del cluster
nome_cluster = None
for nome, index in centroidi_con_nomi.items():
    if index == cluster_utente:
        nome_cluster = nome

# Mostra risultato
if nome_cluster:
    st.markdown(f"🎯 **Il tuo cluster è:** {nome_cluster}")

    # Strategie per ogni cluster
    strategie = {
        "Cacciatori di offerte": [
            "📰 Newsletter settimanale",
            "🎁 Bundle 2+1 snack",
            "🏆 Programma punti",
            "🛒 Offerte in homepage"
        ],
        "Per lover organizzati": [
            "📦 Abbonamenti personalizzati",
            "📧 Reminder riacquisto",
            "🚀 Promo early-access"
        ],
        "Animali affezionati": [
            "💌 Email mensile personalizzata",
            "🎉 Sconti ricorrenze",
            "💎 Up-selling mirato",
            "🧪 Test nuovi prodotti"
        ],
        "Amici animali premium": [
            "💠 Upselling mirato",
            "🎁 Packaging premium",
            "📖 Storytelling via email",
            "🎫 Programma VIP"
        ]
    }

    st.markdown("📌 **Strategie consigliate per questo profilo:**")
    for s in strategie[nome_cluster]:
        st.write(f"- {s}")

