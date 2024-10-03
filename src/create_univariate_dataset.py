import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def create_univariate_dataset_linear(num_samples, noise_level, test_size, random_state):
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

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Print the shapes to verify the split
    print(f"Training set: X_train shape = {X_train.shape}, y_train shape = {y_train.shape}")
    print(f"Testing set: X_test shape = {X_test.shape}, y_test shape = {y_test.shape}")

    return X_train, X_test, y_train, y_test


def create_multiple_dataset_linear(num_samples, num_features, noise_level, test_size, random_state):
    np.random.seed(random_state) #Set random seed

    X_min = np.array([-26, 67, -100, 0,     -10, 0]) #support up t0 6 dimensions in current version.
    X_max = np.array([899, 100, -11, 10000, 10,  3])
    W_all = np.array([81.1, -3.125, 8, 27, -1,   4])
    W = np.zeros(num_features)
    W = W_all[0:num_features]

    b = 158

    y = np.zeros(num_samples)
    X = np.zeros((num_samples, num_features))
    for i in range(num_features):
        X[:,i] = np.random.rand(num_samples, 1).flatten()  * (X_max[i] - X_min[i]) + X_min[i] #Generate random X values between specified min.max ranges
        y += W[i] * X[:,i]
    y += b
    noise = np.random.rand(num_samples,1).flatten() * noise_level #Noise component
    y += noise

    # # Plot the generated dataset
    # rows = int(np.ceil(np.sqrt(num_features)))  # Round up for rows
    # cols = int(np.ceil(num_features / rows))    # Calculate columns based on features
    # fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

    # # Ensure axes is always iterable (even if it's a single plot)
    # if num_features == 1:
    #     axes = [axes]

    # axes = axes.flatten() # Flatten the axes array for easier iteration if num_features is smaller than grid space
    # # Plot each feature against the target y
    # for i in range(num_features):
    #     axes[i].scatter(X[:, i], y, color="blue")
    #     axes[i].set_xlabel(f"Feature {i+1} (X{i+1})")
    #     axes[i].set_ylabel("Target (y)")
    #     axes[i].set_title(f"Feature {i+1} vs Target")

    # for i in range(num_features, len(axes)):
    #     fig.delaxes(axes[i]) # Hide any unused subplots

    # fig.tight_layout() # Adjust layout to avoid overlap
    # plt.show() # Show the plot

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Print the shapes to verify the split
    print(f"Training set: X_train shape = {X_train.shape}, y_train shape = {y_train.shape}")
    print(f"Testing set: X_test shape = {X_test.shape}, y_test shape = {y_test.shape}")

    return X_train, X_test, y_train, y_test



def create_univariate_dataset_poly(n_samples, noise_level, test_size, random_state):
    # Generate random input data 
    X = np.random.rand(n_samples, 1) * 10  # Values between 0 and 10

    # Define the true polynomial relationship
    y = 11.5 * X**3 - 100 * X**2 - 2.5 * X + 4000 +np.random.randn(n_samples, 1) * noise_level

    # Split the dataset into training and testing sets 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Plot the generated data
    plt.scatter(X, y, color='blue')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Generated Polynomial Data (Degree 3)')
    plt.show()

    return X_train, X_test, y_train, y_test

