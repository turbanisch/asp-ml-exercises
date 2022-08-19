import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------ a) ------------------------------------ #

# load data
df = pd.read_csv("https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user", sep = "|")

# ------------------------------------ b) ------------------------------------ #

# print last 10 
df.tail(10)

# print first 25
df.head(25)

# ------------------------------------ c) ------------------------------------ #

# inspect column types
df.dtypes

# ------------------------------------ d) ------------------------------------ #

# count occupations
occupation_counts = df["occupation"].value_counts()

# ------------------------------------ e) ------------------------------------ #

# count unique occupations
df["occupation"].nunique()

# (same as:)
len(occupation_counts)

# most frequent occupation
occupation_counts.head(1)

# ------------------------------------ f) ------------------------------------ #

# sort by index
occupation_counts_byIndex = occupation_counts.sort_index()

# order of rows does not matter for histogram
ax = sns.histplot(data = occupation_counts)
# plt.show()

ax = sns.histplot(data = occupation_counts_byIndex)
# plt.show()

# add x-axis label
ax.set(xlabel = "Number of occurences of each occupation")

# save
plt.savefig("output/occupations.pdf")