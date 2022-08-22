import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import PolynomialFeatures

# ------------------------------------ a) ------------------------------------ #

cancer = load_breast_cancer()
X = cancer["data"] # the actual data
y = cancer["target"] # dependent variable (?)

# ------------------------------------ b) ------------------------------------ #

# extract polynomial features (just guessing because documentation does not explain what "fit" and "transform" do)

poly = PolynomialFeatures(degree = 2, include_bias = False)
output = poly.fit_transform(X)

# number of engineered features
output.shape
# The relevant number here is probably the second one, 495, because the other one is the same as in the original dataset.

# ------------------------------------ c) ------------------------------------ #

# convert to dataframe
colnames = poly.get_feature_names_out()
df = pd.DataFrame(output, columns = colnames)

# add dependent variable
df["y"] = y

# save
df.to_csv("output/polynomials.csv")