# import numpy as np
# import pandas as pd
# from sklearn import svm
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Set up Seaborn for better plot aesthetics
# sns.set_context("notebook", font_scale=1.2)

# # Define the file name
# file_name = "Cupcakes vs Muffins.csv"

# # Load the dataset
# recipes = pd.read_csv(file_name)
# print("File successfully read. Here are the first few rows:")
# print(recipes.head())

# # Correct usage of sns.lmplot() - specifying x and y as keyword arguments
# sns.lmplot(x='Flour', y='Sugar', data=recipes, hue='Type', palette='Set1', fit_reg=False, scatter_kws={"s": 70})

# # Create labels: 0 for Muffins, 1 for Cupcakes
# type_label = np.where(recipes['Type'] == 'Muffin', 0, 1)

# # Extract feature names (excluding the 'Type' column)
# recipe_features = recipes.columns.values[1:].tolist()  # Corrected 'columns' typo
# print("Feature Names:", recipe_features)

# ingredients = recipes[['Flour', 'Sugar']].values
# print("Ingrdiates:", ingredients)


# model = svm.SVC(kernel='linear')
# model.fit(ingredients, type_label)

# w = model.coef_[0]
# print("w: ", w)

# a = -w[0] / w[1]
# xx = np.linspace(30, 60)
# yy = a * xx - (model.intercept_[0]) / w[1]


# # Plot the parallels to the separating hyperplane that pass through the
# b=model.support_vectors_[0]
# yy_down = a * xx + (b[1] - a * b[0])
# b=model.support_vectors_[-1]
# yy_up = a * xx + (b[1] - a * b[0])

# # Plot the hyperplane
# sns.lmplot(x='Flour', y='Sugar', data=recipes, hue='Type', palette='Set1', fit_reg=False, scatter_kws={"s": 70})

# # Plot the line, the points, and the nearest vectors to the plane
# plt.plot(xx, yy, linewidth=2, color='black')
# plt.plot(xx, yy_down, 'k--')
# plt.plot(xx, yy_up, 'k--')

# def muffin_or_cupcake(flour, sugar):
#     if(model.predict([[flour, sugar]]))==0:
#         print("You'll need a muffin.")
#     else:
#         print("You'll need a cupcake.")


# sns.lmplot(x='Flour', y='Sugar', data=recipes, hue='Type', palette='Set1', fit_reg=False, scatter_kws={"s": 70})
# plt.plot(xx, yy, linewidth=2, color='black')

# plt.plot(50, 20, 'yo', markersize='9')        


# muffin_or_cupcake(50, 20)
# muffin_or_cupcake(40, 20)

# plt.tight_layout()
# plt.show()


import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns

# Set up Seaborn for better plot aesthetics
sns.set_context("notebook", font_scale=1.2)

# Define the file name
file_name = "Cupcakes vs Muffins.csv"

# Load the dataset
recipes = pd.read_csv(file_name)
print("File successfully read. Here are the first few rows:")
print(recipes.head())
print("\nShape of the dataset:", recipes.shape)

# Create labels: 0 for Muffins, 1 for Cupcakes
type_label = np.where(recipes['Type'] == 'Muffin', 0, 1)
print("\nType labels:", type_label)

# Extract features
ingredients = recipes[['Flour', 'Sugar']].values
print("\nIngredients (Flour, Sugar):")
print(ingredients)

# Train SVM model
model = svm.SVC(kernel='linear')
model.fit(ingredients, type_label)

# Get model parameters
w = model.coef_[0]
b = model.intercept_[0]
print("\nModel coefficients (w):", w)
print("Model intercept (b):", b)

# Calculate decision boundary
a = -w[0] / w[1]
xx = np.linspace(30, 60)
yy = a * xx - b / w[1]

print("\nDecision boundary slope (a):", a)
print("Decision boundary y-intercept:", -b / w[1])
print("Decision boundary equation: y =", a, "x +", -b / w[1])

# Calculate support vectors
support_vectors = model.support_vectors_
print("\nSupport vectors:")
print(support_vectors)

# Calculate margin lines
sv1 = support_vectors[0]
sv2 = support_vectors[-1]

yy_down = a * xx + (sv1[1] - a * sv1[0])
yy_up = a * xx + (sv2[1] - a * sv2[0])

print("\nLower margin line equation: y =", a, "x +", sv1[1] - a * sv1[0])
print("Upper margin line equation: y =", a, "x +", sv2[1] - a * sv2[0])

# Calculate margin width
margin_width = (sv2[1] - a * sv2[0] - (sv1[1] - a * sv1[0])) / np.sqrt(1 + a**2)
print("\nMargin width:", margin_width)

# Plotting
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Flour', y='Sugar', data=recipes, hue='Type', palette='Set1', s=70)

plt.plot(xx, yy, linewidth=2, color='black', label='Decision Boundary')
plt.plot(xx, yy_down, 'k--', label='Margin')
plt.plot(xx, yy_up, 'k--')

plt.xlabel('Flour')
plt.ylabel('Sugar')
plt.title('Muffin vs Cupcake Classification')
plt.legend()

# Function to predict and explain
def muffin_or_cupcake(flour, sugar):
    point = np.array([flour, sugar])
    decision_value = np.dot(w, point) + b
    prediction = model.predict([point])[0]
    
    print(f"\nPrediction for Flour={flour}, Sugar={sugar}:")
    print(f"Decision value: {decision_value}")
    print(f"Prediction: {'Muffin' if prediction == 0 else 'Cupcake'}")
    
    plt.plot(flour, sugar, 'yo', markersize=10)
    plt.annotate(f'({flour}, {sugar})', (flour, sugar), xytext=(5, 5), textcoords='offset points')

# Make predictions
muffin_or_cupcake(50, 20)
muffin_or_cupcake(40, 20)

plt.tight_layout()
plt.show()

# Print recipe statistics
print("\nRecipe Statistics:")
print(recipes.groupby('Type').agg({'Flour': ['mean', 'min', 'max'], 'Sugar': ['mean', 'min', 'max']}))