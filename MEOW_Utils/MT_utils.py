import copy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class get_bert_element():
    def __init__(self, bertmodel):
        self.bertmodel = bertmodel

    def get_copy_embeddings_layer(self):
        embedding_layer = copy.deepcopy(self.bertmodel.embeddings)
        return embedding_layer

def plot_diagram(H, epoch_num, has_accuracy=False):
    # tensor to float
    H['train_loss'] = [float(i) for i in H['train_loss']]
    if has_accuracy:
        H['train_acc'] = [float(i) for i in H['train_acc']]
    # H['val_loss'] = [float(i) for i in H['val_loss']]
    # H['val_acc'] = [float(i) for i in H['val_acc']]

    # loss
    plt.figure()
    plt.title("Loss")
    plt.xlabel("EPOCH")
    plt.ylabel("Loss")
    plt.plot(H["train_loss"], label="test_loss")
    # plt.plot(H["val_loss"], label="test_loss")
    plt.xticks(np.arange(epoch_num), range(1,epoch_num+1,1))
    plt.show()

    # accuracy
    plt.figure()
    plt.title("Test Accuracy")
    plt.xlabel("EPOCH")
    plt.ylabel("Accuracy")
    plt.plot(H["train_acc"], label="test_acc")
    # plt.plot(H["val_acc"], label="test_acc")
    plt.xticks(np.arange(epoch_num), range(1,epoch_num+1,1))
    plt.show()
