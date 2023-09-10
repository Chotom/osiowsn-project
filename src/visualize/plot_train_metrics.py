import matplotlib.pyplot as plt


def plot_metrics(train_losses, test_losses, test_accuracies):
    # Plot training and testing losses
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
    plt.plot(range(len(test_losses)), test_losses, label='Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Testing Losses')
    plt.legend()

    # Plot testing accuracies
    plt.subplot(1, 2, 2)
    plt.plot(range(len(test_accuracies)), test_accuracies, label='Testing Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Testing Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
