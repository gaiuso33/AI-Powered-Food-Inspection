from sklearn.metrics import confusion_matrix, classification_report
from data_augementation2 import model, val_generator
# Predict on validation set
val_preds = model.predict(val_generator)
val_preds = (val_preds > 0.5).astype("int32")

# Get true labels
true_labels = val_generator.classes

# Confusion Matrix
cm = confusion_matrix(true_labels, val_preds)
print(cm)

# Classification Report
print(classification_report(true_labels, val_preds))
