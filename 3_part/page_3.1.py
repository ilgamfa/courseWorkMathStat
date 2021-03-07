import pandas as pd
import numpy as np


df = pd.read_csv("r3z1.csv")
mean = np.mean(df.X)
print(mean)
