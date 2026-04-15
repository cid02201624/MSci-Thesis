import numpy as np
import matplotlib.pyplot as plt

# Put your values here
cm = np.array([
    [0.94, 0.06],
    [0.30, 0.70]
])

class_names = ["Non-GW", "GW"]

fig, ax = plt.subplots(figsize=(5, 4))

im = ax.imshow(cm, interpolation="nearest", cmap="PuBu")

# Make colorbar match matrix height
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax.set_title("Binary Confusion Matrix (Row-Normalised)", fontsize=14)

tick_marks = np.arange(len(class_names))
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(class_names, fontsize=11)
ax.set_yticklabels(class_names, fontsize=11)
ax.set_xlabel("Predicted label", fontsize=13)
ax.set_ylabel("True label", fontsize=13)

# Grid lines
ax.set_xticks(np.arange(-0.5, len(class_names), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(class_names), 1), minor=True)
ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.8, alpha=0.6)
ax.tick_params(which="minor", bottom=False, left=False)

# Cell text
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j, i, f"{cm[i, j]:.2f}",
            ha="center", va="center",
            fontsize=12,
            color="white" if cm[i, j] > thresh else "black"
        )

plt.tight_layout()
plt.savefig("confusion_matrix.pdf")

# import matplotlib.pyplot as plt
# import numpy as np

# # Input your 4 values here
# top_left = 0.94
# top_right = 0.06
# bottom_left = 0.30
# bottom_right = 0.70

# cm = np.array([
#     [top_left, top_right],
#     [bottom_left, bottom_right]
# ])

# fig, ax = plt.subplots(figsize=(6, 5))

# im = ax.imshow(cm, cmap="PuBu")

# # Axis ticks and labels
# labels = ["Non-GW", "GW"]
# ax.set_xticks([0, 1])
# ax.set_yticks([0, 1])
# ax.set_xticklabels(labels, fontsize=14)
# ax.set_yticklabels(labels, fontsize=14)
# ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.8, alpha=0.6)

# # Axis titles
# ax.set_xlabel("Predicted label", fontsize=16)
# ax.set_ylabel("True label", fontsize=16)
# ax.set_title("Binary Confusion Matrix (Row-Normalised)", fontsize=18, pad=10)

# # Cell annotations
# for i in range(2):
#     for j in range(2):
#         color = "white" if cm[i, j] > 0.5 else "black"
#         ax.text(j, i, f"{cm[i, j]:.2f}",
#                 ha="center", va="center",
#                 fontsize=16, color=color)

# # Colorbar
# plt.colorbar(im, ax=ax)

# plt.tight_layout()
# plt.savefig("confusion_matrix.pdf")