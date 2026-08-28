"""
Downloads the real PneumoniaMNIST dataset and confirms it loaded correctly.
Run this with: python download_pneumonia_data.py
"""

from medmnist import PneumoniaMNIST

train_dataset = PneumoniaMNIST(split='train', download=True)
val_dataset = PneumoniaMNIST(split='val', download=True)
test_dataset = PneumoniaMNIST(split='test', download=True)

print(f"train examples: {len(train_dataset)}")
print(f"val examples:   {len(val_dataset)}")
print(f"test examples:  {len(test_dataset)}")

# peek at one example to confirm shape/labels look right
img, label = train_dataset[0]
print(f"one image type: {type(img)}, label: {label}")