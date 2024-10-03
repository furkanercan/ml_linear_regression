import numpy as np
import matplotlib.pyplot as plt
import math, copy
from create_univariate_dataset import create_multiple_dataset_linear
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def compute_cost (X, y, w, b):
    
    m = X.shape[0] #number of training examples
    cost = 0

    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost += (f_wb_i - y[i])**2

    cost = cost/(2 * m)

    return cost.item()

def compute_gradient (X, y, w, b): #CHECK CORRECTNESS

    m,n = X.shape # number of samples, number of features

    dj_dw = np.zeros(n)
    dj_db = 0
    dj_dw_i = np.zeros(n)

    for i in range (m):
        f_wb = np.dot(X[i], w) + b
        err = f_wb - y[i]

        for j in range (n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]

        dj_db = dj_db + (f_wb - y[i])
    
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db.item() #item: ensure scalar type

def gradient_descent (X, y, w_in, b_in, alpha, num_iters, epsilon, adjust_alpha_enable, cost_function, gradient_function):
    J_history = [] #cost function history
    # p_history = [] #parameter history
    b = b_in
    w = copy.deepcopy(w_in)
    for i in range(num_iters): #do it for a fixed num_iters times, no sense of convergence.
        dj_dw, dj_db = gradient_function(X, y, w, b) #calculate gradient descent
        w_tmp = w - alpha * dj_dw #update weight parameter
        b_tmp = b - alpha * dj_db #update bias parameter

        eval_cost = cost_function(X, y, w_tmp, b_tmp)
        if(i > 1):
            if(eval_cost > J_history[-1]):
                print(f"Adjusting alpha back from {alpha} to {alpha/3}")
                alpha = alpha / 3
                adjust_alpha_enable = 0
                #Don't inherit new w and b params, move on with new alpha in the new iteration.
            elif(adjust_alpha_enable):
                print(f"Adjusting alpha forward from {alpha} to {alpha*3}")
                alpha = alpha * 3
                w = w_tmp
                b = b_tmp
                J_history.append(eval_cost)
                # p_history.append([w, b])
            else:
                # print(f"{eval_cost} <= {J_history[-1]} but adjust_alpha_enable:{adjust_alpha_enable}")
                w = w_tmp
                b = b_tmp
                J_history.append(eval_cost)
                # p_history.append([w, b])
        else:
            w = w_tmp
            b = b_tmp
            J_history.append(eval_cost)
            # p_history.append([w, b])
        
    

        if(i > 2): #Apply convergence
            if(np.abs(J_history[-2] - J_history[-1]) <= epsilon):
                print(f"Convergence at iteration {i}")
                print(f"J_history[-1]: {J_history[-1]}")
                print(f"J_history[-2]: {J_history[-2]}")
                print(f"difference: {J_history[-2] - J_history[-1]}")
                break

        if(i % math.ceil(num_iters/10) == 0):
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw {dj_dw}, dj_db {dj_db:0.3e}, ", 
                  f"w {w}, b {b:0.5e}")
        
    return w, b, J_history#, p_history


num_samples = 1000
num_features = 3
noise_level = 100
test_size = 0.2
random_state = 291192
feature_normalization = 1
adjust_alpha_enable = 1

X_train_original, X_test_original, y_train, y_test = create_multiple_dataset_linear(num_samples, num_features, noise_level, test_size, random_state)

if(feature_normalization):
    mean_X_train = np.zeros(num_features)
    std_X_train = np.zeros(num_features)
    X_train_normalized = np.zeros_like(X_train_original)
    X_test_normalized = np.zeros_like(X_test_original)
    for i in range(num_features):
        mean_X_train[i] = np.mean(X_train_original[:,i])
        std_X_train[i] = np.std(X_train_original[:,i])   
        X_train_normalized[:,i] = (X_train_original[:,i] - mean_X_train[i])/std_X_train[i]
        X_test_normalized[:,i] = (X_test_original[:,i] - mean_X_train[i])/std_X_train[i]
    X_train = X_train_normalized
    X_test = X_test_normalized
else:
    X_train = X_train_original
    X_test = X_test_original

#Initialize parameters
w = np.zeros(num_features)    # Can start from anywhere you want
b = 0                # Can start from anywhere you want
num_iters = 10000    # Number of iterations
tmp_alpha = 0.00004  # Learning rate
epsilon = 0.001

w_final, b_final, J_history = gradient_descent(X_train, y_train, w, b, tmp_alpha, num_iters, epsilon, adjust_alpha_enable, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: {w_final}, {b_final:8.4f}")



### Testing Phase
y_pred = np.dot(X_test, w_final) + b_final
mse = mean_squared_error(y_test,y_pred)
print(f"Mean Squared Error (MSE): {mse}")
r2 = r2_score(y_test,y_pred)
print(f"R² Score: {r2}")

### Plot the generated dataset
rows = int(np.ceil(np.sqrt(num_features)))  # Round up for rows
cols = int(np.ceil(num_features / rows))    # Calculate columns based on features
fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

# Ensure axes is always iterable (even if it's a single plot)
if num_features == 1:
    axes = [axes]

axes = axes.flatten() # Flatten the axes array for easier iteration if num_features is smaller than grid space
# Plot each feature against the target y
for i in range(num_features):
    axes[i].scatter(X_test_original[:,i], y_test, color="blue", label='Actual')
    axes[i].scatter(X_test_original[:,i], y_pred, color="red", label='Predicted')
    axes[i].set_xlabel(f"Feature {i+1} (X_{i+1})")
    axes[i].set_ylabel("Target (y_test/y_pred)")
    axes[i].set_title(f"Actual vs Predicted: Feature {i+1}")

for i in range(num_features, len(axes)):
    fig.delaxes(axes[i]) # Hide any unused subplots

fig.tight_layout() # Adjust layout to avoid overlap
plt.legend()
plt.show() # Show the plot


# # Scatter plot of actual vs. predicted values
# plt.scatter(X_test_original[:,2], y_test, color='blue', label='Actual')
# plt.scatter(X_test_original[:,2], y_pred, color='red', label='Predicted')

# plt.xlabel("X_test (Feature)")
# plt.ylabel("y_test (Target)")
# plt.title("Actual vs. Predicted Values on Testing Set")
# plt.legend()
# plt.show()