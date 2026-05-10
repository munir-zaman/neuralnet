# this script was generated using Anthropic AI (Sonnet 4.6) - munir

import torchvision
import os

dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True)

saved = {}
os.makedirs('mnist_samples', exist_ok=True)

for img, label in dataset:
    if label not in saved:
        img.save(f'mnist_samples/{label}.bmp')
        saved[label] = True
    if len(saved) == 10:
        break