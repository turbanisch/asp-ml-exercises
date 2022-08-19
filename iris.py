import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------ a) ------------------------------------ #

# define column names
colnames = [
    "sepal length (in cm)",
    "sepal width (in cm)",
    "petal length (in cm)",
    "petal width (in cm)",
    "class"
]

# load data
iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", names = colnames)

# ------------------------------------ b) ------------------------------------ #

# create missing values
iris.loc[10:29, "petal length (in cm)"] = None

# ------------------------------------ c) ------------------------------------ #

# replace missing values with 1.0
iris["petal length (in cm)"] = iris["petal length (in cm)"].fillna(1)

# ------------------------------------ d) ------------------------------------ #

# save (without index)
iris.to_csv("output/iris.csv", index = False)

# ------------------------------------ e) ------------------------------------ #

# add ID for each observation and reshape longer
iris["id"] = iris.index
iris_long = iris.melt(id_vars = ["id", "class"], value_vars = colnames[0:4]).sort_values("id")

# plot
sns.catplot(
    data = iris_long,
    x = "class", y = "value",
    col = "variable", col_wrap = 2
)

plt.savefig("output/iris.pdf")

