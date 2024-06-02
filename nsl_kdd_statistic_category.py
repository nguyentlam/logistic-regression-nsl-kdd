import seaborn as sns
import matplotlib.pyplot as plt

# Data
types = ['Normal', 'DOS', 'Probes', 'R2L', 'U2R']
numbers = [67343, 45927, 11656, 995, 52]

# Create bar plot using Seaborn
plt.figure(figsize=(10, 6))
sns.barplot(x=types, y=numbers, palette=['blue'] + sns.color_palette()[1:])
plt.title('Category Network Statistic')
plt.xlabel('Traffic Type')
plt.ylabel('Number of Instances')

plt.savefig('Category_Statistic.png')

# Show plot
plt.show()
