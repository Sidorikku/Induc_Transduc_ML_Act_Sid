import numpy as np
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import accuracy_score

def transductive_learning(train_images_flat, train_labels, test_images_flat, test_labels):
    print("Starting Transductive Learning...")

    # Reduce the dataset size for debugging
    reduced_train_images_flat = train_images_flat[:5000]
    reduced_train_labels = train_labels[:5000]
    reduced_test_images_flat = test_images_flat[:1000]
    reduced_test_labels = test_labels[:1000]

    print("Reduced dataset size for debugging.")
    print(f"Train images shape: {reduced_train_images_flat.shape}")
    print(f"Train labels shape: {reduced_train_labels.shape}")
    print(f"Test images shape: {reduced_test_images_flat.shape}")
    print(f"Test labels shape: {reduced_test_labels.shape}")

    # Create a mix of labeled and unlabeled data
    labeled_ratio = 0.1  # 10% of the data is labeled
    num_labeled = int(len(reduced_train_labels) * labeled_ratio)
    
    # Shuffle the data before labeling
    indices = np.arange(len(reduced_train_labels))
    np.random.shuffle(indices)
    
    labels = np.copy(reduced_train_labels)
    labels[indices[num_labeled:]] = -1  # Set the rest of the data as unlabeled

    print(f"Labeled data count: {np.sum(labels != -1)}")
    print(f"Unlabeled data count: {np.sum(labels == -1)}")
    
    # Train the Label Spreading model
    label_spread_model = LabelSpreading()
    print("Fitting the Label Spreading model...")
    label_spread_model.fit(reduced_train_images_flat, labels)
    
    # Predict the labels for test data
    print("Predicting labels for the test set...")
    predicted_labels = label_spread_model.predict(reduced_test_images_flat)
    
    # Calculate the accuracy
    accuracy = accuracy_score(reduced_test_labels, predicted_labels)
    print(f"Transductive learning accuracy: {accuracy}")
    print("Transductive Learning Completed.")
