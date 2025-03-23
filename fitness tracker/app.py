import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Page Title
st.title("🏃 Personal Fitness Tracker")

# Sidebar Input Section
st.sidebar.header("User Input Parameters")

def user_input_features():
    age = st.sidebar.slider("Age", 18, 70, 25)
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    height = st.sidebar.slider("Height (cm)", 140, 200, 170)
    weight = st.sidebar.slider("Weight (kg)", 40, 120, 70)
    activity_level = st.sidebar.selectbox(
        "Activity Level", ("Sedentary", "Lightly Active", "Moderately Active", "Very Active")
    )
    return {"Age": age, "Gender": gender, "Height": height, "Weight": weight, "Activity Level": activity_level}

input_data = user_input_features()

# Data Processing
data = pd.DataFrame(input_data, index=[0])
data["Gender"] = data["Gender"].apply(lambda x: 1 if x == "Male" else 0)
activity_map = {"Sedentary": 1.2, "Lightly Active": 1.375, "Moderately Active": 1.55, "Very Active": 1.725}
data["Activity Level"] = data["Activity Level"].map(activity_map)

# Calculate BMR & Calorie Requirement
if data["Gender"].values[0] == 1:
    bmr = 10 * data["Weight"].values[0] + 6.25 * data["Height"].values[0] - 5 * data["Age"].values[0] + 5
else:
    bmr = 10 * data["Weight"].values[0] + 6.25 * data["Height"].values[0] - 5 * data["Age"].values[0] - 161

calories_required = bmr * data["Activity Level"].values[0]
st.subheader("🏋️ Estimated Calorie Requirements")
st.write(f"**BMR:** {round(bmr, 2)} kcal/day")
st.write(f"**Daily Calorie Requirement:** {round(calories_required, 2)} kcal/day")

# Fitness Data Simulation for Analysis
n_samples = 200
age_data = np.random.randint(18, 70, n_samples)
weight_data = np.random.randint(50, 100, n_samples)
calories_burned = weight_data * 0.1 * age_data / 30

fitness_df = pd.DataFrame({"Age": age_data, "Weight": weight_data, "Calories Burned": calories_burned})

# Simple Linear Regression (Manual)
X = np.column_stack((np.ones(n_samples), fitness_df["Age"], fitness_df["Weight"]))
y = fitness_df["Calories Burned"].values
theta = np.linalg.inv(X.T @ X) @ X.T @ y

# Prediction Calculation
age_input = data["Age"].values[0]
weight_input = data["Weight"].values[0]
prediction = theta[0] + theta[1] * age_input + theta[2] * weight_input
st.subheader("🔥 Predicted Calories Burned")
st.write(f"**Estimated Calories Burned:** {round(prediction, 2)} kcal")

# Visualization with Matplotlib
st.subheader("📊 Data Visualization")
fig, ax = plt.subplots()
scatter = ax.scatter(fitness_df["Age"], fitness_df["Calories Burned"], c=fitness_df["Weight"], cmap="coolwarm", label="Calories Burned")
legend1 = ax.legend(*scatter.legend_elements(), title="Weight (kg)")
ax.add_artist(legend1)
plt.xlabel("Age")
plt.ylabel("Calories Burned")
st.pyplot(fig)
