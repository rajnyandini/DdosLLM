import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../scripts/processed_data/carpet_bombing_dataset.csv')

plt.figure(figsize=(10, 6))
sns.kdeplot(data=df[df['label'] == 0], x='min_pkt_len', label='MAWI (Benign)', fill=True)
sns.kdeplot(data=df[df['label'] == 1], x='min_pkt_len', label='CIC (Attack)', fill=True)
plt.title("Numerical Feature Gap: Benign vs Attack")
plt.legend()
plt.show()