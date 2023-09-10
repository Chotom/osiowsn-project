import matplotlib.pyplot as plt
import numpy as np


def plot_example_images(dataset, class_indices, class_names):
    num_classes = len(class_indices)
    num_examples_per_class = 1
    example_images = {class_name: [] for class_name in class_indices}

    for data, label in dataset:
        if label in class_indices:
            example_images[label].append(data)

    plt.figure(figsize=(5*num_classes, 5))

    for i, class_name in enumerate(class_indices):
        for j in range(num_examples_per_class):
            plt.subplot(1, num_classes * num_examples_per_class, i * num_examples_per_class + j + 1)
            plt.title(class_names[i])
            plt.axis('off')
            plt.imshow(np.transpose(example_images[class_name][j].numpy(), (1, 2, 0)))
    plt.show()
