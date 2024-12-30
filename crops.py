#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Data from the classification report
true_labels = np.concatenate([
    [i] * support for i, support in enumerate([169, 75, 23, 7, 217, 17, 23, 9, 9, 23, 380, 209])
])

# Adjust probabilities so they sum to 1
probabilities = [0.13, 0.07, 0.03, 0.01, 0.17, 0.01, 0.03, 0.01, 0.01, 0.03, 0.38, 0.22]
probabilities = np.array(probabilities) / sum(probabilities)

predicted_labels = np.random.choice(
    np.arange(12), len(true_labels), p=probabilities
)

# Confusion Matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=np.arange(12))
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(12))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title("Confusion Matrix for Crop Recommendation")
plt.show()

# Classification Report
report = classification_report(
    true_labels, predicted_labels, labels=np.arange(12),
    target_names=[f"Crop {i}" for i in range(12)]
)
print("Classification Report:\\n", report)

# Bar Chart for Macro and Weighted Averages
precision = [0.53, 0.24, 0.00, 0.00, 0.51, 0.00, 0.27, 0.00, 0.00, 0.17, 0.57, 0.39]
recall = [0.32, 0.07, 0.00, 0.00, 0.65, 0.00, 0.13, 0.00, 0.00, 0.04, 0.76, 0.44]
f1_score = [0.40, 0.10, 0.00, 0.00, 0.57, 0.00, 0.18, 0.00, 0.00, 0.07, 0.65, 0.41]
crop_labels = [f"Crop {i}" for i in range(12)]

x = np.arange(len(crop_labels))
width = 0.25
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width, precision, width, label='Precision', color='blue')
ax.bar(x, recall, width, label='Recall', color='green')
ax.bar(x + width, f1_score, width, label='F1-Score', color='orange')

ax.set_xlabel("Crop Classes")
ax.set_ylabel("Performance Metrics")
ax.set_title("Performance Metrics by Crop Class")
ax.set_xticks(x)
ax.set_xticklabels(crop_labels, rotation=45)
ax.legend()

plt.tight_layout()
plt.show()


# In[ ]:




