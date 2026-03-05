import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_hermite
import math
import time

st.set_page_config(page_title="Oscilador 1D Dinámico", layout="wide")
st.title("Laboratorio 1D: Dinámica y Teorema de Ehrenfest")

# ==========================================
# 0. MEMORIA DEL SISTEMA
# ==========================================
if 'tiempo_t' not in st.session_state:
    st.session_state.tiempo_t = 0.0

# ==========================================
# 1. MOTOR FÍSICO 1D
# ==========================================
@st.cache_data
def generar_base_1d(limit=5.0, points=400, omega=1.0):
    x = np.linspace(-limit, limit, points)
    dx = x[1] - x[0]
    
    def psi_n(coord, n):
        xi = np.sqrt(omega) * coord
        norm = (omega / np.pi)**0.25 / math.sqrt((2**n) * math.factorial(n))
        return norm * np.exp(-0.5 * xi**2) * eval_hermite(n, xi)
    
    Psi_base = [psi_n(x, i) for i in range(3)]
    E_vals = [omega * (i + 0.5) for i in range(3)]
    V_x = 0.5 * omega**2 * x**2
    
    return x, dx, Psi_base, E_vals, V_x

x, dx, Psi_base, E_vals, V_x = generar_base_1d()

# ==========================================
# 2. PANEL DE CONTROL (BARRA LATERAL)
# ==========================================
with st.sidebar:
    st.header("Control de Tiempo")
    animar = st.toggle("Reproducir Animación", value=True)
    velocidad = st.slider("Velocidad ($dt$)", 0.01, 0.5, 0.08, step=0.01)
    
    st.markdown("---")
    st.header("Vector de Estado $|\Psi\\rangle$")
    st.markdown("Ajusta el módulo $|c_n|$ y la fase $\\phi_n$ de cada estado.")
    
    col_amp, col_fase = st.columns(2)
    with col_amp:
        a0 = st.slider(r"$|c_0|$", 0.0, 1.0, 1.0, 0.1)
        a1 = st.slider(r"$|c_1|$", 0.0, 1.0, 1.0, 0.1)
        a2 = st.slider(r"$|c_2|$", 0.0, 1.0, 0.0, 0.1)
    with col_fase:
        f0 = st.slider(r"$\phi_0$ (rad)", 0.0, 2*np.pi, 0.0, 0.1)
        f1 = st.slider(r"$\phi_1$ (rad)", 0.0, 2*np.pi, 0.0, 0.1)
        f2 = st.slider(r"$\phi_2$ (rad)", 0.0, 2*np.pi, 0.0, 0.1)

# ==========================================
# 3. ÁLGEBRA Y CÁLCULOS CUÁNTICOS
# ==========================================
# Normalización e inclusión de las fases complejas
norm_total = np.sqrt(a0**2 + a1**2 + a2**2)
if norm_total == 0:
    c0, c1, c2 = 1.0, 0.0, 0.0
else:
    c0 = (a0/norm_total) * np.exp(1j * f0)
    c1 = (a1/norm_total) * np.exp(1j * f1)
    c2 = (a2/norm_total) * np.exp(1j * f2)

E_media = (np.abs(c0)**2)*E_vals[0] + (np.abs(c1)**2)*E_vals[1] + (np.abs(c2)**2)*E_vals[2]

t_actual = st.session_state.tiempo_t

# Función de onda total dependiente del tiempo
Psi_t = (c0 * Psi_base[0] * np.exp(-1j * E_vals[0] * t_actual) + 
         c1 * Psi_base[1] * np.exp(-1j * E_vals[1] * t_actual) + 
         c2 * Psi_base[2] * np.exp(-1j * E_vals[2] * t_actual))

Prob_t = np.abs(Psi_t)**2

# Integral del valor esperado de la posición: <x> = integral( x * |Psi|^2 dx )
exp_x = np.sum(x * Prob_t) * dx

# ==========================================
# 4. INTERFAZ GRÁFICA Y FÓRMULAS
# ==========================================
# Fórmulas dinámicas en la parte superior
st.markdown("### Formalismo Matemático del Estado Actual")
eq_estado = r"|\Psi(0)\rangle = "
if np.abs(c0) > 0: eq_estado += rf"({np.abs(c0):.2f} e^{{{f0:.1f}i}})|0\rangle + "
if np.abs(c1) > 0: eq_estado += rf"({np.abs(c1):.2f} e^{{{f1:.1f}i}})|1\rangle + "
if np.abs(c2) > 0: eq_estado += rf"({np.abs(c2):.2f} e^{{{f2:.1f}i}})|2\rangle "
eq_estado = eq_estado.rstrip("+ ")

st.latex(eq_estado)
st.latex(rf"\langle E \rangle = \sum |c_n|^2 E_n = {E_media:.3f} \, \hbar\omega")

st.markdown("---")

col1, col2 = st.columns([2.5, 1])

with col1:
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Potencial
    ax.plot(x, V_x, 'k-', lw=1.5, label=r'$V(x)$')
    # Densidad de Probabilidad
    ax.fill_between(x, E_media, E_media + Prob_t, color='steelblue', alpha=0.6)
    ax.plot(x, E_media + Prob_t, color='navy', lw=2, label=r'$|\Psi(x,t)|^2$')
    
    # Marcar el Valor Esperado <x> (Teorema de Ehrenfest)
    ax.plot(exp_x, E_media, 'ro', markersize=8, label=r'$\langle x \rangle(t)$')
    # Línea que une <x> con la curva para visualizar mejor el centro de masa
    ax.vlines(exp_x, E_media, E_media + np.interp(exp_x, x, Prob_t), color='red', linestyle='--', alpha=0.7)
    
    ax.set_ylim(0, max(E_vals) + 1.5)
    ax.set_xlim(-4.5, 4.5)
    ax.set_xlabel(r'Posición $x$')
    ax.set_ylabel(r'Energía / Probabilidad')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    
    st.pyplot(fig)

with col2:
    st.subheader("Observables Físicos")
    st.markdown(f"**Tiempo $(t)$:** `{t_actual:.2f}`")
    st.markdown(f"**Posición $\\langle x \\rangle$:** `{exp_x:.3f}`")
    
    st.markdown("**Probabilidades ($P_n = |c_n|^2$):**")
    # Barra de progreso visual para las probabilidades
    st.progress(float(np.abs(c0)**2), text=f"P0: {np.abs(c0)**2 * 100:.1f}%")
    st.progress(float(np.abs(c1)**2), text=f"P1: {np.abs(c1)**2 * 100:.1f}%")
    st.progress(float(np.abs(c2)**2), text=f"P2: {np.abs(c2)**2 * 100:.1f}%")

# ==========================================
# 5. BUCLE DE ANIMACIÓN
# ==========================================
if animar:
    st.session_state.tiempo_t += velocidad
    time.sleep(0.04)
    st.rerun()