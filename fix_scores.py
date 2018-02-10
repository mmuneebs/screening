import numpy as np
import pandas as pd
import os

scores = np.loadtxt('output.txt')
df = pd.read_csv('stage1_sample_submission.csv')
df['Probability'] = scores
df.to_csv('subp.csv', index=False)
