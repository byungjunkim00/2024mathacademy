!pip install plotly

import streamlit as st
import numpy as np
import plotly.graph_objs as go

def runge_kutta(x0, y0, k1, k2, k3, t_max, h=0.1):
    n = int(np.ceil(t_max / h))
    t = np.linspace(0, t_max, n)
    x = np.zeros(n)
    y = np.zeros(n)
    x[0], y[0] = x0, y0

    for i in range(1, n):
        xi, yi = x[i-1], y[i-1]

        kx1 = h * (k1 * xi - k2 * xi * yi)
        ky1 = h * (k2 * xi * yi - k3 * yi)

        kx2 = h * (k1 * (xi + kx1/2) - k2 * (xi + kx1/2) * (yi + ky1/2))
        ky2 = h * (k2 * (xi + kx1/2) * (yi + ky1/2) - k3 * (yi + ky1/2))

        kx3 = h * (k1 * (xi + kx2/2) - k2 * (xi + kx2/2) * (yi + ky2/2))
        ky3 = h * (k2 * (xi + kx2/2) * (yi + ky2/2) - k3 * (yi + ky2/2))

        kx4 = h * (k1 * (xi + kx3) - k2 * (xi + kx3) * (yi + ky3))
        ky4 = h * (k2 * (xi + kx3) * (yi + ky3) - k3 * (yi + ky3))

        x[i] = xi + (kx1 + 2*kx2 + 2*kx3 + kx4) / 6
        y[i] = yi + (ky1 + 2*ky2 + 2*ky3 + ky4) / 6

    return t, x, y

st.title("Differential Equations Solver")

st.markdown("## System of Equations")
st.latex(r"""
\frac{dX}{dt} = k_1 \cdot X - k_2 \cdot X \cdot Y
""")
st.latex(r"""
\frac{dY}{dt} = k_2 \cdot X \cdot Y - k_3 \cdot Y
""")

st.sidebar.header("Parameters")
x0 = st.sidebar.selectbox("Initial X", range(1, 10), index=0)
y0 = st.sidebar.selectbox("Initial Y", range(1, 10), index=0)
k1 = st.sidebar.selectbox("k₁", range(1, 10), index=0)
k2 = st.sidebar.selectbox("k₂", range(1, 10), index=0)
k3 = st.sidebar.selectbox("k₃", range(1, 10), index=0)
t_max = st.sidebar.number_input("Max t", min_value=1, value=40, step=1)

if st.sidebar.button("Run"):
    t, x, y = runge_kutta(x0, y0, k1, k2, k3, t_max)
    
    st.markdown("## Solution")
    trace_x = go.Scatter(x=t, y=x, mode='lines', name='X(t)')
    trace_y = go.Scatter(x=t, y=y, mode='lines', name='Y(t)')
    
    layout = go.Layout(
        title="Solution to the System of Differential Equations",
        xaxis={'title': 't'},
        yaxis={'title': 'X(t), Y(t)'}
    )
    
    fig = go.Figure(data=[trace_x, trace_y], layout=layout)
    st.plotly_chart(fig)
