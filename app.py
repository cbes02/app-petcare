import streamlit as st
import numpy as np
from sklearn.cluster import KMeans

st.title("🐾 Scopri il tuo profilo cliente PetCare!")

st.write("Inserisci la **spesa media mensile** e il **numero di acquisti** per scoprire a quale cluster appartieni e quali strategie sono più adatte a te.")

spesa = st.number_input("💰 Spesa media mensile (€)", min_value=0.0, value=100.0, step=10.0)
acquisti = st.number_input("🛍️ Numero di acquisti mensili", min_value=0, value=5, step=1)

np.random.seed(42)

cluster1 = np.column_stack((np.random.normal(loc=100, scale=5, size=50), np.random.normal(loc=10, scale=2, size=50)))
cluster2 = np.column_stack((np.random.normal(loc=115, scale=5, size=50), np.random.normal(loc=13, scale=2, size=50)))
cluster3 = np.column_stack((np.random.normal(loc=130, scale=5, size=50), np.random.normal(loc=5, scale=2, size=50)))
cluster4 = np.column_stack((np.random.normal(loc=160, scale=5, size=50), np.random.normal(loc=4, scale=2, size=50)))

X = np.vstack((cluster1, cluster2, cluster3, cluster4))
X[:, 1] = np.clip(X[:, 1], a_min=1, a_max=None)

nuovo_cliente = np.array([[spesa, acquisti]])

kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(X)
cluster_assegnato = kmeans.predict(nuovo_cliente)[0]

strategie = {
    0: {"nome": "🎯 Cacciatori di offerte", "strategie": ["📰 Newsletter settimanale", "🎁 Bundle 2+1 snack", "🏆 Programma punti", "🛒 Offerte in homepage"]},
    1: {"nome": "📦 Per lover organizzati", "strategie": ["📬 Abbonamenti", "⏰ Reminder riordino", "🚀 Promo early-access"]},
    2: {"nome": "🐕‍🦺 Animali affezionati", "strategie": ["📅 Email mensile", "🎉 Sconti ricorrenze", "🔝 Up-selling mirato", "🧪 Test prodotti"]},
    3: {"nome": "👑 Amici animali premium", "strategie": ["💎 Upselling mirato", "🎁 Packaging premium", "📖 Storytelling via email", "🎟️ Programma VIP"]}
}

st.subheader(f"🎯 Il tuo cluster è: **{strategie[cluster_assegnato]['nome']}**")

st.write("📌 Strategie consigliate per questo profilo:")
for s in strategie[cluster_assegnato]["strategie"]:
    st.markdown(f"- {s}")
