import numpy as np
import matplotlib.pyplot as plt
import math
from create_univariate_dataset import create_univariate_dataset
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def compute_cost (x, y, w, b):
    
    m = x.shape[0] #number of training examples
    cost_sum = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i])**2
        cost_sum = cost_sum + cost     
    cost_final = cost_sum/(2 * m)

    return cost_final.item()

def compute_gradient (x, y, w, b):

    m = x.shape[0]

    dj_dw = 0
    dj_db = 0

    for i in range (m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = (f_wb - y[i])
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw.item(), dj_db.item() #item: ensure scalar type

def gradient_descent (x, y, w_in, b_in, alpha, num_iters, epsilon, adjust_alpha_enable, cost_function, gradient_function):
    J_history = [] #cost function history
    p_history = [] #parameter history
    b = b_in
    w = w_in
    for i in range(num_iters): #do it for a fixed num_iters times, no sense of convergence.
        dj_dw, dj_db = gradient_function(x, y, w, b) #calculate gradient descent
        w_tmp = w - alpha * dj_dw #update weight parameter
        b_tmp = b - alpha * dj_db #update bias parameter

        eval_cost = cost_function(x, y, w_tmp, b_tmp)
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
                p_history.append([w, b])
            else:
                # print(f"{eval_cost} <= {J_history[-1]} but adjust_alpha_enable:{adjust_alpha_enable}")
                w = w_tmp
                b = b_tmp
                J_history.append(eval_cost)
                p_history.append([w, b])
        else:
            w = w_tmp
            b = b_tmp
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])
        
    

        if(i > 2): #Apply convergence
            if(np.abs(J_history[-2] - J_history[-1]) <= epsilon):
                print(f"Convergence at iteration {i}")
                print(f"J_history[-1]: {J_history[-1]}")
                print(f"J_history[-2]: {J_history[-2]}")
                print(f"difference: {J_history[-2] - J_history[-1]}")
                break

        if(i % math.ceil(num_iters/10) == 0):
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw {dj_dw:0.3e}, dj_db {dj_db:0.3e}, ", 
                  f"w {w:0.3e}, b {b:0.5e}")
        
    return w, b, J_history, p_history


num_samples = 1000
noise_level = 2
test_size = 0.2
random_state = 291192
feature_normalization = 1
adjust_alpha_enable = 1

X_train_original, X_test_original, y_train, y_test = create_univariate_dataset(num_samples, noise_level, test_size, random_state)

if(feature_normalization):
    mean_X_train = np.mean(X_train_original)
    std_X_train = np.std(X_train_original)   
    X_train_normalized = (X_train_original - mean_X_train)/std_X_train
    X_test_normalized = (X_test_original - mean_X_train)/std_X_train
    X_train = X_train_normalized
    X_test = X_test_normalized
else:
    X_train = X_train_original
    X_test = X_test_original

#Initialize parameters
w = 0             # Can start from anywhere you want
b = 0             # Can start from anywhere you want
num_iters = 10000 # Number of iterations
tmp_alpha = 0.00004  # Learning rate
epsilon = 0.001

w_final, b_final, J_history, p_history = gradient_descent(X_train, y_train, w, b, tmp_alpha, num_iters, epsilon, adjust_alpha_enable, compute_cost, compute_gradient)

print(f"(w,b) found by gradient descent: {w_final:8.4f}, {b_final:8.4f}")



### Testing Phase
y_pred = w_final * X_test + b_final
mse = mean_squared_error(y_test,y_pred)
print(f"Mean Squared Error (MSE): {mse}")
r2 = r2_score(y_test,y_pred)
print(f"R² Score: {r2}")



# Scatter plot of actual vs. predicted values
plt.scatter(X_test_original, y_test, color='blue', label='Actual')
plt.scatter(X_test_original, y_pred, color='red', label='Predicted')

plt.xlabel("X_test (Feature)")
plt.ylabel("y_test (Target)")
plt.title("Actual vs. Predicted Values on Testing Set")
plt.legend()
plt.show()

### TODO:
### Regularization to avoid overfitting  y (probably doesn't apply to linear regression)
### Next up: 02 Univariate polynomial regression
### Next up: 03 Multiple linear regression
### Next up: 04 Multiple polynomial regression