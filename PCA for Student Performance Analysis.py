# Step 1: Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# Step 2: Define data with more subjects
data = {
    'Math Score': [80, 75, 90, 85, 70, 78, 92, 88, 72, 81],
    'Science Score': [85, 78, 92, 88, 75, 80, 95, 90, 78, 84],
    'English Score': [90, 80, 88, 91, 82, 85, 87, 89, 80, 87],
    'History Score': [70, 85, 75, 90, 88, 80, 82, 76, 78, 84],
    'Art Score': [95, 90, 85, 88, 92, 89, 94, 90, 88, 91]
}

# Step 3: Creating a DataFrame
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Step 4: Performing PCA and fitting
pca = PCA(n_components=1)
principal_component = pca.fit_transform(df)

# Step 5: Adding principal component scores to DataFrame
df['Principal_Component'] = principal_component
print("\nDataFrame with Principal Component:")
print(df)

# Step 6: Visualization
math_score = df["Math Score"]
science_score = df["Science Score"]
english_score = df["English Score"]
history_score = df["History Score"]
art_score = df["Art Score"]
pc_component = df["Principal_Component"]

fig = plt.figure(figsize=(14, 6))

# 3D plot of original scores with enhanced colors and markers
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(math_score, science_score, english_score, c='cyan', marker='o', label='Scores')
ax1.set_xlabel('Math Score', color='blue')
ax1.set_ylabel('Science Score', color='green')
ax1.set_zlabel('English Score', color='red')
ax1.set_title('Original Scores', fontsize=14)
ax1.legend()

# 2D plot of the first principal component with enhanced colors
ax2 = fig.add_subplot(122)
ax2.scatter(principal_component, np.zeros_like(pc_component), c='orange', marker='x', label='Principal Component')
ax2.set_xlabel('Principal Component', color='darkorange')
ax2.set_title('First Principal Component', fontsize=14)
ax2.axhline(0, color='grey', lw=0.5, ls='--')  # Adding a horizontal line at y=0
ax2.legend()

plt.tight_layout()
plt.show()
