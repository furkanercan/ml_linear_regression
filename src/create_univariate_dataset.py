import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def create_univariate_dataset(num_samples, noise_level, test_size, random_state):
    np.random.seed(random_state) #Set random seed

    X = np.random.rand(num_samples, 1) * 100 #Generate random X values between 0 and 100

    noise = np.random.rand(num_samples,1) * noise_level #Noise component

    y = 12.5 * X + 22 + noise

    # Plot the generated dataset
    plt.scatter(X, y, color="blue")
    plt.xlabel("Feature (X)")
    plt.ylabel("Target (y)")
    plt.title("Generated Synthetic Data")
    # plt.show()

    # Split the dataset into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Print the shapes to verify the split
    print(f"Training set: X_train shape = {X_train.shape}, y_train shape = {y_train.shape}")
    print(f"Testing set: X_test shape = {X_test.shape}, y_test shape = {y_test.shape}")

    return X_train, X_test, y_train, y_test