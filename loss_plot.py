import json

import matplotlib.pyplot as plt
import numpy as np

with open("mlp.json") as f:
	data_mlp = json.load(f)

with open("no_mlp.json") as f:
	data_no_mlp = json.load(f)

mlp_time = np.array(data_mlp.get("time")[1:])
mlp_time = mlp_time - mlp_time[0]
mlp_loss = data_mlp.get("loss")

no_mlp_time = np.array(data_no_mlp.get("time")[1:])
no_mlp_time = no_mlp_time - no_mlp_time[0]
no_mlp_loss = data_no_mlp.get("loss")

plt.semilogy(mlp_loss, label="Original")
plt.semilogy(no_mlp_loss, label="Alternative")
plt.title("Training loss")
plt.xlabel("Step")
plt.ylabel("Log loss")
plt.legend()
plt.show()

"""
plt.plot(mlp_loss)
plt.title("Training loss")
plt.xlabel("Step")
plt.ylabel("Loss")

ax = plt.gca()

# [left, bottom, width, height] relative to the parent axes (0 to 1)
axins = ax.inset_axes([0.6, 0.6, 0.35, 0.35])
axins.plot(mlp_loss)

axins.set_xlim(0, 50)
axins.set_xticklabels([])
axins.set_yticklabels([])

ax.indicate_inset_zoom(axins, edgecolor="black")

plt.show()
"""
