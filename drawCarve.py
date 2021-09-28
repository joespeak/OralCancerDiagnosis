import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"/save111.csv")
print(data)

plt.figure(figsize=(6, 4.5))
plt.plot(data.index + 1, data["acc"], label="all_acc")
plt.plot(data.index+1, data["acc0"], label="acc_normal")
plt.plot(data.index+1, data["acc1"], label="acc_ossc")

plt.axis([0, 120, 0.89, 1])
plt.legend()
plt.show()