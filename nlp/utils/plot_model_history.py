import matplotlib.pyplot as plt


def plot(model_trained):
    # Plot accuracy
    plt.plot(model_trained.history['acc'])
    plt.plot(model_trained.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot loss
    plt.plot(model_trained.history['loss'])
    plt.plot(model_trained.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()
