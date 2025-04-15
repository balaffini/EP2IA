import numpy as np
import matplotlib.pyplot as plt
import string
import sys

def plot_confusion_matrix(file_path, letter):
    # Load the confusion matrix from the text file
    conf_matrix = np.loadtxt(file_path)
    
    # Get the index of the letter
    labels = list(string.ascii_uppercase)
    letter_index = labels.index(letter.upper())

    # Extract True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN)
    TP = conf_matrix[letter_index, letter_index]
    FP = conf_matrix[:, letter_index].sum() - TP
    FN = conf_matrix[letter_index, :].sum() - TP
    TN = conf_matrix.sum() - (TP + FP + FN)

    # Create a new confusion matrix for VP, FP, VN, FN
    new_conf_matrix = np.array([[TP, FN],
                                [FP, TN]])

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(new_conf_matrix, interpolation='nearest', cmap=plt.cm.PiYG, vmin=0, vmax=new_conf_matrix.max())
    plt.title('CLASSIFICAÇÃO DO MODELO')
    plt.colorbar()

    # Add labels to the axes
    plt.xticks([0, 1], ['Predito:', 'Predito:'])
    plt.yticks([0, 1], ['Real:', 'Real:'])

    # Add text annotations with VP, FN, FP, VN
    labels_annotations = [['VP', 'FN'], ['FP', 'VN']]
    thresh = new_conf_matrix.max() / 2.
    for i, j in np.ndindex(new_conf_matrix.shape):
        plt.text(j, i, f'{labels_annotations[i][j]}\n{int(new_conf_matrix[i, j])}', 
                 horizontalalignment="center",
                 color="white" if new_conf_matrix[i, j] > thresh else "black")

    plt.ylabel('REAL')
    plt.xlabel('PREDITO')
    plt.tight_layout()

    # Save the plot to a file
    output_plot_path = file_path.replace('.txt', f'_{letter}_plot.png')
    plt.savefig(output_plot_path)
    print(f'Plot saved as: {output_plot_path}')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python plot_confusion_matrix_vp_fn.py <path_to_confusion_matrix_file> <letter>")
    else:
        file_path = sys.argv[1]
        letter = sys.argv[2]
        plot_confusion_matrix(file_path, letter)
