from tensorflow.keras import layers, models

def inductive_learning(train_images, train_labels, test_images, test_labels):
    print("Starting Inductive Learning...")
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=5)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Inductive learning test accuracy: {test_acc}")
    print("Inductive Learning Completed.")
