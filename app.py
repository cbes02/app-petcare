import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Simulazione dati per 4 cluster
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
            "Spesa media mensile (€)": round(xi, 2),
            "Acquisti mensili": round(yi, 2),
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

# Palette colori personalizzata
palette = {
    "Cacciatori di offerte": "deeppink",
    "Pet lover organizzati": "sandybrown",
    "Animali affezionati": "dimgray",
    "Amici animali premium": "mediumseagreen"
}

# (Rimozione della visualizzazione del grafico)
