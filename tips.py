import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------ a) ------------------------------------ #

# load built-in dataset
tips = sns.load_dataset("tips")

# ------------------------------------ b) ------------------------------------ #

# inspect which levels "day" takes on
tips["day"].cat.categories

# create dict for replacement
weekday_dict = {
    "Thur": "Thursday",
    "Fri": "Friday",
    "Sat": "Saturday",
    "Sun": "Sunday"
}

# replace
tips["day"] = tips["day"].replace(weekday_dict)

# check results
tips["day"].cat.categories

# ------------------------------------ c) ------------------------------------ #

# create plot and save axis object
ax = sns.relplot(
    data = tips, 
    x = "total_bill", y = "tip",
    col = "sex", hue = "day",
    kind = "scatter"
)

# set axis labels
ax.set(xlabel = "Total bill (USD)", ylabel = "Tip (USD)")

# save plot
plt.savefig("output/tips.pdf")
