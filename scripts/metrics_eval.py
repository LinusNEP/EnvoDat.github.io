import pandas as pd
import matplotlib.pyplot as plt

results_file = '.../results.csv'  # Path to the training results CSV file

def load_training_results(file_path):
    df = pd.read_csv(file_path)
    return df

# Training and validation losses
def plot_losses(df):
    # Extract relevant columns
    train_box_loss = df['train/box_loss']
    train_cls_loss = df['train/cls_loss']
    train_dfl_loss = df['train/dfl_loss']
    val_box_loss = df['val/box_loss']
    val_cls_loss = df['val/cls_loss']
    val_dfl_loss = df['val/dfl_loss']
    epochs = df['epoch']

    # Training losses
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_box_loss, label='Box Loss', color='blue')
    plt.plot(epochs, train_cls_loss, label='Classification Loss', color='orange')
    plt.plot(epochs, train_dfl_loss, label='DFL Loss', color='green')
    plt.title('Training Losses Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Validation losses
    plt.subplot(2, 1, 2)
    plt.plot(epochs, val_box_loss, label='Validation Box Loss', color='blue')
    plt.plot(epochs, val_cls_loss, label='Validation Classification Loss', color='orange')
    plt.plot(epochs, val_dfl_loss, label='Validation DFL Loss', color='green')
    plt.title('Validation Losses Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Validation metrics
def plot_validation_metrics(df):
    precision = df['metrics/precision(B)']
    recall = df['metrics/recall(B)']
    map_value = df['metrics/mAP50(B)']
    epochs = df['epoch']

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, precision, label='Precision', color='blue')
    plt.plot(epochs, recall, label='Recall', color='orange')
    plt.plot(epochs, map_value, label='mAP', color='green')
    plt.title('Validation Metrics Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# Learning rate over time
def plot_learning_rate(df):
    # Assuming you want to plot the learning rate from the first group (pg0)
    learning_rate = df['lr/pg0']
    epochs = df['epoch']

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, learning_rate, label='Learning Rate (pg0)', color='purple')
    plt.title('Learning Rate Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    df = load_training_results(results_file)
    plot_losses(df)
    plot_validation_metrics(df)
    plot_learning_rate(df)

if __name__ == "__main__":
    main()

