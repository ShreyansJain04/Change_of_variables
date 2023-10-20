# Import necessary libraries
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, lambdify


def change_of_variables(jacobian_matrix, function, x):
    x = np.array(x)
    transformed_x = np.dot(jacobian_matrix, x)
    transformed_y = function(*transformed_x)
    return transformed_x, transformed_y

st.title("Change of Variables Using Jacobian Matrix for Functions")


st.header("Define Jacobian Matrix")
matrix_rows = st.slider("Number of Rows", min_value=1, max_value=5, value=2)
matrix_cols = st.slider("Number of Columns", min_value=1, max_value=5, value=2)

jacobian_matrix = []
for i in range(matrix_rows):
    row = []
    for j in range(matrix_cols):
        row.append(st.number_input(f"Jacobian[{i}][{j}]", value=0.0, key=f"jacobian_{i}_{j}"))
    jacobian_matrix.append(row)

jacobian_matrix = np.array(jacobian_matrix)


st.header("Input Function")
function_str = st.text_input("Enter the mathematical function (e.g., 'x**2 + y**2'):")

x, y = symbols('x y')
user_function = lambdify((x, y), function_str, 'numpy')


x_values = np.linspace(-10, 10, 400)
y_values = np.linspace(-10, 10, 400)
X, Y = np.meshgrid(x_values, y_values)
Z = user_function(X, Y)


st.header("Input Vector")
input_vector = []
for i in range(matrix_cols):
    input_vector.append(st.number_input(f"Input Vector[{i}]", key=f"input_{i}"))

input_vector = np.array(input_vector)


transformed_x, transformed_y = change_of_variables(jacobian_matrix, user_function, input_vector)


st.header("Result")
st.write("Transformed Vector (x):", transformed_x)
st.write("Transformed Vector (y):", transformed_y)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.quiver(0, 0, input_vector[0], input_vector[1], angles='xy', scale_units='xy', scale=1, color='b', label='Original Vector')
ax1.quiver(0, 0, transformed_x[0], transformed_x[1], angles='xy', scale_units='xy', scale=1, color='r', label='Transformed Vector')
ax1.set_xlim([0, max(input_vector[0], transformed_x[0])])
ax1.set_ylim([0, max(input_vector[1], transformed_x[1])])
ax1.set_aspect('equal')
ax1.legend()
ax1.set_xlabel('X')
ax1.set_ylabel('Y')

contour = ax2.contourf(X, Y, Z, cmap='viridis')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
plt.colorbar(contour, ax=ax2)

st.pyplot(fig)
