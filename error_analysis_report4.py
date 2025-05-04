from sklearn.metrics import confusion_matrix, classification_report
from data_augementation2 import model, val_generator

val_preds = model.predict(val_generator)
val_preds = (val_preds > 0.5).astype("int32")

true_labels = val_generator.classes

cm = confusion_matrix(true_labels, val_preds)
print(cm)

print(classification_report(true_labels, val_preds))
