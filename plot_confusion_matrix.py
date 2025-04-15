import numpy as np
import matplotlib.pyplot as plt
import string

def plot_confusion_matrix(file_path):
    # Load the confusion matrix from the text file
    conf_matrix = np.loadtxt(file_path)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    # Add labels to the axes
    labels = list(string.ascii_uppercase)
    tick_marks = np.arange(conf_matrix.shape[0])
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    # Add text annotations
    thresh = conf_matrix.max() / 2.
    for i, j in np.ndindex(conf_matrix.shape):
        plt.text(j, i, int(conf_matrix[i, j]), 
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    # Save the plot to a file
    output_plot_path = file_path.replace('.txt', '_plot.png')
    plt.savefig(output_plot_path)
    print(f'Plot saved as: {output_plot_path}')

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python plot_confusion_matrix.py <path_to_confusion_matrix_file>")
    else:
        plot_confusion_matrix(sys.argv[1])
