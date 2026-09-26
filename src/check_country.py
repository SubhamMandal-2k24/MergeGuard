import pandas as pd

data_dir = r"C:\Users\ASUS\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset"

s1 = pd.read_csv(data_dir + r"\train\train_source1.tsv", sep="\t", dtype=str)
s2 = pd.read_csv(data_dir + r"\train\train_source2.tsv", sep="\t", dtype=str)
s3 = pd.read_csv(data_dir + r"\train\train_source3.tsv", sep="\t", dtype=str)

print(s1["country"].value_counts())
print(s2["country"].value_counts())
print(s3["country"].value_counts())