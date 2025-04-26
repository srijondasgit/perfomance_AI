import pandas as pd

# Load the raw lines
with open("train.txt", "r") as f:
    lines = f.readlines()

# Split each line into inputs and labels
data = [line.strip().split("-> Label:") for line in lines]
df = pd.DataFrame(data, columns=["Input", "Label"])

# Optional: strip extra whitespace
df["Input"] = df["Input"].str.strip()
df["Label"] = df["Label"].str.strip()

# Display the first few rows
df.head()
