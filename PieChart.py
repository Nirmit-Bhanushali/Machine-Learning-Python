import matplotlib.pyplot as plt
import numpy as np
# Pie Charts
y=np.array([35, 25, 25, 15])
# add labels
mylabels=["Mumbai", "Nagpur", "Pune", "Amravati"]
# add explode
myexplode=[0, 0, 0.3, 0]
# add colors
mycolors=["y", "#56AFC4", "b", "#4CAF50"]

plt.pie(y, labels=mylabels, explode=myexplode, shadow=True, colors=mycolors)
plt.legend(title="Cities")
plt.show()