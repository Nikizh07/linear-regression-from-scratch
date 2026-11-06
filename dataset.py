import numpy as np
import pandas as pd

# Reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# Generate study hours between 0 and 12
hours = np.random.uniform(0, 12, n_samples)

# Generate scores
# True relationship:
# Score = 15 + 7 * Hours + Noise
noise = np.random.normal(0, 8, n_samples)
scores = 15 + 7 * hours + noise

# Keep scores within 0-100
scores = np.clip(scores, 0, 100)

# Round values
hours = np.round(hours, 2)
scores = np.round(scores, 2)

# Create DataFrame
df = pd.DataFrame({
    "Hours": hours,
    "Score": scores
})

# Save to CSV
df.to_csv("student_scores.csv", index=False)

print(df.head())
print(f"\nDataset saved with {len(df)} samples.")