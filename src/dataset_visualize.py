import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
print(os.getcwd())
os.chdir('..')
print(os.getcwd())

GNAD_df = pd.read_csv(r'C:\Users\Hany\PycharmProjects\NLP-Assignment\data\articles.csv', header=None, sep=';', quotechar="'", names=['label', 'text'])
GNAD_df.head()
plt.xticks(rotation='vertical')
plt.hist(GNAD_df['label'])
plt.show()