import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Palette colori e descrizioni
palette = {
    "Cacciatori di offerte": "deeppink",
    "Pet lover organizzati": "sandybrown",
    "Animali affezionati": "dimgray",
    "Amici animali premium": "mediumseagreen"
}

strategie = {
    "Cacciatori di offerte": [
        "📰 Newsletter settimanale",
        "🎁 Bundle 2+1 snack",
        "🏆 Programma punti",
        "🛒 Offerte in homepage"
    ],
    "Pet lover organizzati": [
        "📦 Abbonamenti personalizzati",
        "⏰ Reminder riacquisto",
        "✨ Promo early-access"
    ],
    "Animali affezionati": [
        "📧 Email mensile personalizzata",
        "🎉 Sconti ricorrenze",
        "💎 Up-selling mirato",
        "🧪 Test nuovi prodotti"
    ],
    "Amici animali premium": [
        "💎 Upselling mirato",
        "🎁 Packaging premium",
        "📖 Storytelling via email",
        "🎫 Programma VIP"
    ]
}

# Simulazione dati coerente con il tuo grafico
np.random.seed(42)
cluster_specs = {
    "Cacciatori di offerte": {"mean": (100, 10), "std": (5, 1), "count": 50},
    "Pet lover organizzati": {"mean": (115, 13), "std": (5, 1), "count": 50},
    "Animali affezionati": {"mean": (130, 5), "std": (5, 1), "count": 50},
    "Amici animali premium": {"mean": (160, 4), "std": (5, 1), "count": 50},
}

data = []
for cluster, specs in cluster_specs.items():
    x = np.random.normal(specs["mean"][0], specs["std"][0], specs["count"])
    y = np.random.normal(specs["mean"][1], specs["std"][1], specs["count"])
    for xi, yi in zip(x, y):
        data.append({
            "Spesa": xi,
            "Acquisti": yi,
            "Cluster": cluster
        })

df = pd.DataFrame(data)

# Centroidi stimati
centroids = {
    "Cacciatori di offerte": (100, 10),
    "Pet lover organizzati": (115, 13),
    "Animali affezionati": (130, 5),
    "Amici animali premium": (160, 4)
}

# Funzione per assegnare cluster
def assegna_cluster(x, y):
    distanze = {k: np.sqrt((x - cx) ** 2 + (y - cy) ** 2) for k, (cx, cy) in centroids.items()}
    return min(distanze, key=distanze.get)

# Streamlit UI
st.title("\U0001F43E Scopri il tuo profilo cliente PetCare!")
st.markdown("""
Inserisci la **spesa media mensile** e il **numero di acquisti** per scoprire a quale cluster appartieni e quali strategie sono più adatte a te.
""")

spesa = st.number_input("\U0001F4B0 Spesa media mensile (€)", min_value=0.0, value=100.0, step=1.0)
acquisti = st.number_input("\U0001F6CD Numero di acquisti mensili", min_value=0, value=5, step=1)

profilo = assegna_cluster(spesa, acquisti)
st.markdown(f"### 🎯 Il tuo cluster è: **{profilo}**")
st.markdown("**\U0001F4CC Strategie consigliate per questo profilo:**")
for s in strategie[profilo]:
    st.write(f"- {s}")

# Grafico
fig, ax = plt.subplots(figsize=(10, 6))
for cluster in df["Cluster"].unique():
    subset = df[df["Cluster"] == cluster]
    ax.scatter(subset["Spesa"], subset["Acquisti"], label=cluster, color=palette[cluster], edgecolors='black')

for cluster, (x, y) in centroids.items():
    ax.scatter(x, y, color=palette[cluster], marker='X', s=200, edgecolors='black', label=None)

# Nuovo punto utente
ax.scatter(spesa, acquisti, color="gray", marker='*', s=200, edgecolors='black', label="Nuovo Cliente")

ax.set_xlabel("Spesa media mensile (€)")
ax.set_ylabel("Numero di acquisti mensili")
ax.set_title("Segmentazione clienti con K-Means")
ax.legend()
st.pyplot(fig)
