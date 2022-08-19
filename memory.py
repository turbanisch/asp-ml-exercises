import pandas as pd

# ------------------------------------ a) ------------------------------------ #

# load data
df = pd.read_csv("https://query.data.world/s/wsjbxdqhw6z6izgdxijv5p2lfqh7gx")

# ------------------------------------ b) ------------------------------------ #

# inspect
df.info()
df.info(memory_usage = "deep")

# according to the docs: A value of ‘deep’ is equivalent to “True with deep introspection”.
# i.e., setting "deep" yields accurate memory usage whereas otherwise it is just estimated based on dtype and row numbers

# ------------------------------------ c) ------------------------------------ #

# create copy, "object" only
df_objects = df.select_dtypes(include = "object")

# ------------------------------------ d) ------------------------------------ #

# inspect summary
df_summary = df_objects.describe().loc[["unique", "count"]].transpose()

# compute ratio of unique values to the number of non-missing observations for each variable
df_summary["unique_count_ratio"] = df_summary["unique"] / df_summary["count"]

# identify 5 variables with lowest unique-to-count ratio
df_summary.sort_values("unique_count_ratio").head(5)

# ------------------------------------ e) ------------------------------------ #

# make sense in terms of memory usage?
# according to the docs: If the number of categories approaches the length of the data, the Categorical will use nearly the same or more memory than an equivalent object dtype representation. 
# https://pandas.pydata.org/docs/user_guide/categorical.html#categorical-memory
# unclear where the cutoff is, probably higher than 50%?

# ------------------------------------ f) ------------------------------------ #

# conversion to category does not make sense if unique-count ratio is very high
df_summary.sort_values("unique_count_ratio", ascending=False).head(5)

# apart from "completion" and "rf_umpire_id", all variables have ratios < 26%
# for these, a conversion to category could be beneficial 
# however, "completion" and "rf_umpire_id" have so few observations that they probably don't matter much

# convert all object to category types
list_str_obj_cols = df.columns[df.dtypes == "object"].tolist()
df.select_dtypes(include = "object")

for str_obj_col in list_str_obj_cols:
    df[str_obj_col] = df[str_obj_col].astype("category")

# ------------------------------------ g) ------------------------------------ #

# check new memory usage
df.info()
df.info(memory_usage = "deep")

# new memory usage is lower: 158.7 MB
                
# ------------------------------------ h) ------------------------------------ #

# Conversion could have been sped up by setting the dtype or converter in pandas.read_csv()
# However, we would have to know in advance which columns to convert

# ------------------------------------- i ------------------------------------ #

# subset numeric columns
df_numeric = df.select_dtypes(include = "number")

# save as csv and feather
df_numeric.to_csv("output/memory.csv")
df_numeric.to_feather("output/memory.feather")

# on my disk, the csv file is more than twice as big than the feather file (52.5 MB vs. 21.8 MB), due to compression algorithms in feather
# but more impressive is the speed of saving (apparently the main advantage of feather?)