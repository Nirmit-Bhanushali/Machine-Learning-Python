import pandas as pd
import numpy as np

# Sample data
df = pd.DataFrame({'Income': [1000, 5000, 10000, 50000]})

# Log Transformation
df['Log_Income'] = np.log(df['Income'])

print(df)