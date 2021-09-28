import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# df = pd.DataFrame({"normal": [0.8, 0.9], "ossc": [0.8, 0.6]})
# fig = sns.barplot(data=df)

models = ["ResNet", "Vgg19", "RepVgg", "MobileV3Small", "MobileV3Large", "EfficientNet-B7"]
acc = [[0.944, 0.876], [0.94, 0.9], [0.932, 0.9], [0.868, 0.927], [0.932, 0.923], [0.966, 0.911]]
acc0 = [0.944, 0.94, 0.932, 0.868, 0.966, 0.932]
acc1 = [0.876, 0.9, 0.9, 0.927, 0.911, 0.923]

bar_width = 0.2
x_1 = list(range(len(models)))
xax = [i+bar_width/2 for i in x_1]
x_2 = [i+bar_width for i in x_1]

plt.figure(figsize=(7, 6))

plt.bar(range(len(models)), acc0, width=bar_width, label="normal_acc")
plt.bar(x_2, acc1, width=bar_width, label="ossc_acc")
plt.legend(fontsize=5, loc="upper right")

plt.xticks(xax, models, rotation=90, fontsize=5)

plt.show()
#plt.savefig(r"classAccGraphic.pdf")
