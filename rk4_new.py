import streamlit as st
import numpy as np
import plotly.graph_objects as go

def runge_kutta(f, y0, t):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0

    for i in range(1, n):
        h = t[i] - t[i-1]
        k1 = f(t[i-1], y[i-1])
        k2 = f(t[i-1] + h/2, y[i-1] + k1*h/2)
        k3 = f(t[i-1] + h/2, y[i-1] + k2*h/2)
        k4 = f(t[i-1] + h, y[i-1] + k3*h)

        y[i] = y[i-1] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

    return y

st.title("Differential Equations Solver")

# Inputs
x0 = st.slider("X (Initial Value)", 1, 9, 1)
y0 = st.slider("Y (Initial Value)", 1, 9, 1)
k1 = st.slider("k₁", 1, 9, 1)
k2 = st.slider("k₂", 1, 9, 1)
k3 = st.slider("k₃", 1, 9, 1)
tMax = st.slider("t range", 10, 100, 40)

# Define the differential equations
def f(t, vars):
    x, y = vars
    dxdt = k1 * x - k2 * x * y
    dydt = k2 * x * y - k3 * y
    return np.array([dxdt, dydt])

# Solve the equations
t = np.arange(0, tMax, 0.1)
solution = runge_kutta(f, [x0, y0], t)

# Extract the solutions
x_values = solution[:, 0]
y_values = solution[:, 1]

# Plot the results
fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=x_values, mode='lines', name='X'))
fig.add_trace(go.Scatter(x=t, y=y_values, mode='lines', name='Y'))

fig.update_layout(
    title="Solution to the System of Differential Equations",
    xaxis_title="t",
    yaxis_title="Value",
)

st.plotly_chart(fig)

st.write("This app simulates a system of differential equations using the Runge-Kutta method.")
