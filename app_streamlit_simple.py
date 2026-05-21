import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import io
import base64
import os
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Descente de Gradient - Version Simple",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration matplotlib
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.origin'] = 'lower'
plt.rcParams['image.cmap'] = 'magma_r'

def plot_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"

def sauvegarder(fig, x, y, a_crt, b_crt, a_reel, b_reel, sig_noise, m, eqm_history, a_init, b_init, eta, fois):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dossier = os.path.join(os.path.dirname(__file__), 'sauvegardes', timestamp)
    os.makedirs(dossier, exist_ok=True)

    fig.savefig(os.path.join(dossier, f'vue_ensemble_{timestamp}.png'), dpi=150, bbox_inches='tight')

    results_df = pd.DataFrame({'x': x, 'y': y, 'y_estime': a_crt * x + b_crt})
    results_df.to_csv(os.path.join(dossier, f'donnees_{timestamp}.csv'), index=False)

    eqm_df = pd.DataFrame({'iteration': range(1, len(eqm_history)+1), 'eqm': eqm_history})
    eqm_df.to_csv(os.path.join(dossier, f'evolution_eqm_{timestamp}.csv'), index=False)

    amelioration = (1 - eqm_history[-1] / eqm_history[0]) * 100
    contenu = f"""=== PARAMÈTRES DU CALCUL - {timestamp} ===

PARAMÈTRES DE GÉNÉRATION:
   a réel = {a_reel}
   b réel = {b_reel}
   Bruit (σ) = {sig_noise}
   Nombre de points = {m}

PARAMÈTRES DE DESCENTE:
   a initial = {a_init}
   b initial = {b_init}
   Taux d'apprentissage (η) = {eta}
   Itérations = {fois}

RÉSULTATS OBTENUS:
   a final = {a_crt:.6f}
   b final = {b_crt:.6f}
   EQM initiale = {eqm_history[0]:.8f}
   EQM finale = {eqm_history[-1]:.8f}
   Amélioration = {amelioration:.4f}%
"""
    with open(os.path.join(dossier, f'parametres_{timestamp}.txt'), 'w', encoding='utf-8') as f:
        f.write(contenu)

    return timestamp

# Interface principale
st.title("📊 Descente de Gradient")
st.markdown("Application interactive de descente de gradient")

# Sidebar
st.sidebar.title("⚙️ Paramètres")

st.sidebar.subheader("🎲 Génération des données")
a_reel    = st.sidebar.number_input("a réel",           min_value=0.0,   max_value=2.0,  value=0.7,  step=0.01)
b_reel    = st.sidebar.number_input("b réel",           min_value=0.0,   max_value=1.0,  value=0.3,  step=0.01)
sig_noise = st.sidebar.number_input("Bruit (σ)",        min_value=0.001, max_value=0.1,  value=0.01, step=0.001, format="%.3f")
m         = st.sidebar.number_input("Nombre de points", min_value=10,    max_value=100,  value=30,   step=1)

st.sidebar.subheader("🔬 Descente de gradient")
a_init = st.sidebar.number_input("a initial",                min_value=-2.0, max_value=2.0, value=2.0, step=0.01)
b_init = st.sidebar.number_input("b initial",                min_value=-2.0, max_value=2.0, value=2.0, step=0.01)
eta    = st.sidebar.number_input("Taux d'apprentissage (η)", min_value=0.01, max_value=1.0, value=0.5, step=0.01)
fois   = st.sidebar.number_input("Itérations",               min_value=10,   max_value=100, value=30,  step=1)

m    = int(m)
fois = int(fois)

# -----------------------------
# Calcul en live (pas de bouton)
# -----------------------------
with st.spinner("Mise à jour..."):

    np.random.seed(0)
    x = np.random.uniform(low=0, high=1, size=m)
    y = a_reel * x + b_reel + stats.norm.rvs(size=x.size) * sig_noise
    dx = np.linspace(0, 1, 100)

    Na, Nb = 500, 500
    tab_a = np.linspace(-1, 2, Na)
    tab_b = np.linspace(-1, 2, Nb)
    image = np.zeros((Na, Nb))

    for i in range(Na):
        for j in range(Nb):
            y_chap = tab_a[i] * x + tab_b[j]
            image[i, j] = sum((y_chap - y)**2) / m

    a_crt = a_init
    b_crt = b_init
    tab_a_crt = np.zeros(fois)
    tab_b_crt = np.zeros(fois)
    eqm_history = []

    for i in range(fois):
        y_chap = a_crt * x + b_crt
        dCpara = (2/m) * sum(x * (y_chap - y))
        dCparb = (2/m) * np.sum(y_chap - y)
        a_crt -= eta * dCpara
        b_crt -= eta * dCparb
        tab_a_crt[i] = a_crt
        tab_b_crt[i] = b_crt
        eqm_history.append(sum((y_chap - y)**2) / m)

    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    fig.suptitle("Descente de Gradient - Vue d'ensemble", fontsize=16, fontweight='bold')

    axes[0, 0].plot(x, y, 'o', label='Données')
    axes[0, 0].plot(dx, a_reel * dx + b_reel, label='Objectif')
    axes[0, 0].set_xlabel('Valeurs de x')
    axes[0, 0].set_ylabel('Valeurs de y')
    axes[0, 0].legend()
    axes[0, 0].grid()
    axes[0, 0].set_title('Données générées')

    im2 = axes[0, 1].imshow(image.T, vmin=0, vmax=0.5, extent=[-1, 2, -1, 2], aspect='auto')
    axes[0, 1].plot(a_reel, b_reel, '*', markersize=15, label='Valeurs réelles')
    axes[0, 1].set_xlabel('a')
    axes[0, 1].set_ylabel('b')
    fig.colorbar(im2, ax=axes[0, 1], label='EQM')
    axes[0, 1].legend()
    axes[0, 1].set_title("Carte d'EQM")

    im3 = axes[1, 0].imshow(image.T, vmin=0, vmax=0.5, extent=[-1, 2, -1, 2], aspect='auto')
    axes[1, 0].plot(a_reel, b_reel, '*', markersize=15, label='Valeurs réelles')
    axes[1, 0].plot(a_init, b_init, 'go', markersize=10, label='Point de départ')
    axes[1, 0].plot(tab_a_crt, tab_b_crt, '.-r', linewidth=2, markersize=4, label='Trajectoire')
    axes[1, 0].plot(a_crt, b_crt, 'r*', markersize=15, label='Point final')
    axes[1, 0].set_xlabel('a')
    axes[1, 0].set_ylabel('b')
    fig.colorbar(im3, ax=axes[1, 0], label='EQM')
    axes[1, 0].legend()
    axes[1, 0].set_title('Descente de Gradient')

    axes[1, 1].plot(eqm_history, 'b-', linewidth=2)
    axes[1, 1].set_xlabel('Itération')
    axes[1, 1].set_ylabel('EQM')
    axes[1, 1].set_title("Évolution de l'EQM")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_img = plot_to_image(fig)

# -----------------------------
# Affichage
# -----------------------------
col_fig, col_data = st.columns([3, 1])

with col_fig:
    st.image(fig_img, caption="Les 4 figures regroupées", use_container_width=True)

with col_data:
    st.subheader("📥 Paramètres")
    st.write(f"- **a réel** = {a_reel}")
    st.write(f"- **b réel** = {b_reel}")
    st.write(f"- **Bruit** = {sig_noise}")
    st.write(f"- **Points** = {m}")
    st.markdown("---")
    st.subheader("📤 Résultats")
    st.write(f"- **a final** = {a_crt:.4f}")
    st.write(f"- **b final** = {b_crt:.4f}")
    st.write(f"- **EQM initiale** = {eqm_history[0]:.6f}")
    st.write(f"- **EQM finale** = {eqm_history[-1]:.6f}")

st.markdown("---")

# -----------------------------
# Boutons export et sauvegarde
# -----------------------------
results_df = pd.DataFrame({'x': x, 'y': y, 'y_estime': a_crt * x + b_crt})
eqm_df     = pd.DataFrame({'iteration': range(1, len(eqm_history)+1), 'eqm': eqm_history})
amelioration = (1 - eqm_history[-1] / eqm_history[0]) * 100
timestamp_now = datetime.now().strftime('%Y%m%d_%H%M%S')
txt_contenu = f"""=== PARAMÈTRES DU CALCUL - {timestamp_now} ===

PARAMÈTRES DE GÉNÉRATION:
   a réel = {a_reel}
   b réel = {b_reel}
   Bruit (σ) = {sig_noise}
   Nombre de points = {m}

PARAMÈTRES DE DESCENTE:
   a initial = {a_init}
   b initial = {b_init}
   Taux d'apprentissage (η) = {eta}
   Itérations = {fois}

RÉSULTATS OBTENUS:
   a final = {a_crt:.6f}
   b final = {b_crt:.6f}
   EQM initiale = {eqm_history[0]:.8f}
   EQM finale = {eqm_history[-1]:.8f}
   Amélioration = {amelioration:.4f}%
"""

col_save, col_e1, col_e2, col_e3 = st.columns(4)

with col_save:
    if st.button("💾 Sauvegarder ces résultats"):
        timestamp = sauvegarder(
            fig, x, y, a_crt, b_crt,
            a_reel, b_reel, sig_noise, m,
            eqm_history, a_init, b_init, eta, fois
        )
        st.toast(f"💾 Sauvegarde effectuée dans : sauvegardes/{timestamp}", icon="✅")

with col_e1:
    st.download_button(
        label="📥 CSV données",
        data=results_df.to_csv(index=False),
        file_name=f"donnees_{timestamp_now}.csv",
        mime="text/csv"
    )
with col_e2:
    st.download_button(
        label="📥 CSV EQM",
        data=eqm_df.to_csv(index=False),
        file_name=f"evolution_eqm_{timestamp_now}.csv",
        mime="text/csv"
    )
with col_e3:
    st.download_button(
        label="📥 TXT paramètres",
        data=txt_contenu,
        file_name=f"parametres_{timestamp_now}.txt",
        mime="text/plain"
    )

# Footer
st.markdown("---")
st.markdown("📝 **Application de descente de gradient**")
st.markdown("🔬 **Algorithme de descente de gradient pour régression linéaire**")

# Commande pour lancer l'application
# python -m streamlit run app_streamlit_simple.py