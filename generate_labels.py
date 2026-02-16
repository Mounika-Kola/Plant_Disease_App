import os, json

train_dir = "PlantVillage/train"   # adjust if needed
class_names = sorted(os.listdir(train_dir))

with open("class_labels.json", "w") as f:
    json.dump(class_names, f)

print("✅ class_labels.json created with", len(class_names), "classes")