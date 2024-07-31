from data_loader import load_data
from inductive_learning import inductive_learning
from transductive_learning import transductive_learning

def main():
    train_images, train_labels, test_images, test_labels, train_images_flat, test_images_flat = load_data()
    inductive_learning(train_images, train_labels, test_images, test_labels)
    transductive_learning(train_images_flat, train_labels, test_images_flat, test_labels)

if __name__ == "__main__":
    main()
