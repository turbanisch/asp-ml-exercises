import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ------------------------------------ a) ------------------------------------ #

# load data
df = pd.read_csv("data/olympics.csv", index_col = "id")

# summary statistics
df.describe()

# drop score?
# depends on question. For now, just note that values are much higher and likely generated from the other vars, so it might not contain additional information

# ------------------------------------ b) ------------------------------------ #

# scale to unit variance
# scale data
scaler = StandardScaler()
scaler.fit(df)
df_scaled = scaler.transform(df)

# double-check standard deviation
# pd.describe() could be used to assert that all variables now have unit variance if this were still a pandas dataframe

# ------------------------------------ c) ------------------------------------ #

# fit plain-vanilla PCA

pca = PCA(random_state=42)
pca.fit(df_scaled)

# get loadings
loadings = pd.DataFrame(pca.components_)

# ------------------------------------ d) ------------------------------------ #

# inspect cumulative explained variance
explained_variance = pd.DataFrame(pca.explained_variance_ratio_, columns = ["Explained Variance"])
explained_variance["cumulative"] = explained_variance["Explained Variance"].cumsum()

# 6 components are needed to explain at least 90% of the variation