import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Read CSV file
df = pd.read_csv("test_predictions_binary.csv")

# Extract true and predicted labels
y_true = df["true_class"]
y_pred = df["pred_class_aux"]

# Labels and display names
labels = [0, 1, 2]
class_names = ["Noise", "Glitch", "GW"]

# Row-normalised confusion matrix
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels,
    normalize="true"
)

# Plot
fig, ax = plt.subplots(figsize=(6, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(cmap="Blues", values_format=".2f", ax=ax)

plt.title("Row-Normalised Confusion Matrix")

# Save to PDF
plt.savefig("confusion_matrix_normalised.pdf",
            format="pdf",
            bbox_inches="tight")

plt.show()