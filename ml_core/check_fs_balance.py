import csv

file_path = "e:/fyp/givingUpVersion_v2/data/raw/fs_dataset_clean.csv"
label_counts = {"0": 0, "1": 0}

with open(file_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        label = row['label'].strip()
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1

print(f"Label counts in {file_path}:")
for label, count in label_counts.items():
    print(f"Label {label}: {count}")

total = sum(label_counts.values())
if total > 0:
    print(f"Positive ratio: {label_counts.get('1',0)/total:.2%}")
    print(f"Negative ratio: {label_counts.get('0',0)/total:.2%}")
else:
    print("No samples found.")
