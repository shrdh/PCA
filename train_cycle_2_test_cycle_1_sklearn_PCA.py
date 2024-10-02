# Importing Libaries
# Importing Libaries

# Importing Libaries

import numpy as np
import tensorflow as tf
from scipy.fft import fft, fftfreq
import os

import matplotlib.cm as cm
import pandas as pd
from scipy.stats import poisson
import matplotlib.pyplot as plt
import scipy as scipy
from scipy import stats
import scipy.optimize as opt
import math
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error,root_mean_squared_error

from scipy.optimize import curve_fit

tf.get_logger().setLevel('ERROR')

# folder_cycle1 = r'D:\Downloads\Sensor_1_ESR_cycling-20240213T204739Z-001\Sensor_1_ESR_cycling\sensor1_esr_temp_cycle_1'
# folder_cycle2 =  r'D:\Downloads\sensor_1_ESR_cycle_2'
folder_cycle1=  r'D:\Downloads\sensor1_esr_temp_cycle_1'
#folder_cycle2= r'D:\Downloads\sensor_1_ESR_cycle_2-20240810T185524Z-001\sensor_1_ESR_cycle_2'
folder_cycle2= r'D:\Downloads\sensor_1_ESR_cycle_2-20240811T005415Z-001\sensor_1_ESR_cycle_2'

# Get a list of all the files in the folders (excluding the PARAMS file)
cycle1_files = os.listdir(folder_cycle1)
cycle1_files = [f for f in cycle1_files if "PARAMS" not in f]
cycle2_files = os.listdir(folder_cycle2)
cycle2_files = [f for f in cycle2_files if "PARAMS" not in f]

# Defining Parameters
s1 = np.array([[0.0,1.0,0.0],
    [1.0,0.0,1.0],
    [0.0,1.0,0.0]])

s2 = np.array([[0.0,-1.0j,0.0],
    [1.0j,0.0,-1.0j],
    [0.0,1.0j,0.0]])

s3 = np.array([[1.0,0.0,0.0],
    [0.0,0.0,0.0],
    [0.0,0.0,-1.0]])

spin1 = (1.0/np.sqrt(2.0))*s1
spin2 = (1.0/np.sqrt(2.0))*s2
spin3=s3


spin1 = tf.constant(spin1, dtype = 'complex128')
spin2 = tf.constant(spin2, dtype = 'complex128')
spin3 = tf.constant(spin3, dtype = 'complex128')

# a=tf.constant(-7.86851953723355e-05,dtype='float64')# Linear Regression
# b= tf.constant(2.870665858002803,dtype='float64') # Linear Regression
b= tf.constant( 2.87068615576284,dtype='float64') # Grad search cycle 2 
a=tf.constant(-7.723607188481802e-05, dtype='float64') # Grad search cycle 2 

c=tf.constant( -4.3478260869566193e-07,dtype='float64')
d=tf.constant(0.005185511627906974,dtype='float64')#Literature Value



v = tf.constant(0.00, dtype = 'float64')    # ~N(0,sigma_v)
w = tf.constant(0.00, dtype = 'float64')    # ~N(0,sigma_w)

P_0 = tf.constant(1e-4, dtype = 'float64')
P = tf.constant(0.18, dtype = 'float64')
alpha= tf.constant(14.52e-3, dtype = 'float64')
I = tf.eye(3,dtype = 'complex128')
    
    
    
def getD(T, P):
    D = a * T + b + alpha * (P_0 - P_0)
    E = c * T + d + w
    return D, E

def H(D, E):
    Ham = tf.complex(D * (tf.math.real(spin3 @ spin3 - 2 / 3 * I)) + E * (tf.math.real(spin1 @ spin1 - spin2 @ spin2)),
                    D * (tf.math.imag(spin3 @ spin3 - 2 / 3 * I)) + E * (tf.math.imag(spin1 @ spin1 - spin2 @ spin2)))
    return Ham


@tf.autograph.experimental.do_not_convert
@tf.function
def getP_k(T, P):
    D, E = getD(T, P)
    Ham = H(D, E)
    eigenvalues = tf.linalg.eigvals(Ham)
    return eigenvalues


@tf.function
def bilorentzian(x, T, P):
    eigenvalues = getP_k(T, P)
    x0 = tf.cast(eigenvalues[1] - eigenvalues[2], tf.float64)
    x01 = tf.cast(eigenvalues[0] - eigenvalues[2], tf.float64)
    x = tf.cast(x, tf.float64)
    a = tf.cast( 47.64938163757324, tf.float64)  # cycle2
    gamma = tf.cast(0.004283152054995298, tf.float64)  #cycle2
    
    return a * gamma**2 / ((x - x0)**2 + gamma**2) + a * gamma**2 / ((x - x01)**2 + gamma**2)

def _get_vals(T, P):
    timespace = np.linspace(start_frequency_cycle2, end_frequency_cycle2, num=N)
    timespace = tf.cast(timespace, 'float64')
    vals = bilorentzian(timespace, T, P)
    return tf.reshape(vals, [N, 1])



# Reading Data and taking everything that can be changed
delimiter = "\t"
variable_names = ["Frequency", "Intensity1", "Intensity2"]   


# Cycle 2
test_data=[]
temperatures_cycle2 = [-30.0, -20.0, -10.0,  0.0, 10.0,  20.0, 30.0,  40.0, 50.0,40.0, 30.0,  20.0, 10.0, 0.0, -10.0, -20.0, -30.0]
num_files_per_temp_cycle2 = 2
Frequency_cycle2 = None
# Process each group of 20 files
for i in range(0, len(cycle2_files), num_files_per_temp_cycle2):
    files_group_cycle2 = cycle2_files[i:i+num_files_per_temp_cycle2]
    temp_cycle2 = temperatures_cycle2[i//num_files_per_temp_cycle2]  # Get the corresponding temperature for this group
    T = tf.constant(temp_cycle2, dtype=tf.float64)
    ratios_cycle2 = np.array([])

   
    for file in files_group_cycle2:
        data_cycle2 = pd.read_csv(os.path.join(folder_cycle2, file), delimiter=delimiter, header=None, names=variable_names)

        ratio_cycle2 = np.divide(data_cycle2['Intensity2'], data_cycle2['Intensity1'])
        if ratios_cycle2.size == 0:
            ratios_cycle2 = np.array([ratio_cycle2])
        else:
            ratios_cycle2 = np.vstack((ratios_cycle2, [ratio_cycle2]))  # Add ratio to the numpy array

    avg_intensity_cycle2 = np.mean(ratios_cycle2, axis=0)
    if Frequency_cycle2 is None:
        Frequency_cycle2 = data_cycle2['Frequency'].values
        # Assuming Frequency is in Hz
        Frequency_GHz_cycle2 = Frequency_cycle2 / 1e9
        start_frequency_cycle2 = np.min(Frequency_cycle2)/1e9

    end_frequency_cycle2 = np.max(Frequency_cycle2)/1e9

    N = Frequency_cycle2.shape[0]
    dt = np.round((end_frequency_cycle2 - start_frequency_cycle2) / N, 4)

    timespace = np.linspace(start_frequency_cycle2, end_frequency_cycle2, num=N)
    sim_val = _get_vals(T, P)
    noise_sample_cycle2= avg_intensity_cycle2[np.where(np.abs(timespace)<2.85)[0]] 
    noise_mean_cycle2 = np.mean(noise_sample_cycle2)
    avg_intensity_cycle2 = avg_intensity_cycle2 - noise_mean_cycle2
    avg_intensity_cycle2 = np.max(sim_val)*( avg_intensity_cycle2)/(np.max(avg_intensity_cycle2))
    noise_sample_cycle2= avg_intensity_cycle2[np.where(np.abs(timespace)<2.85)[0]]
    std_noise_cycle2=np.std(noise_sample_cycle2)
    test_data.append(avg_intensity_cycle2)
    




all_data_test_2D = np.array(test_data)

# Cycle 1



all_data = []
all_temperatures = []
all_roots = []
mt_list, mt_orig_list, valt_list, valt_orig_list = [[] for _ in range(4)]

# Reading Data and taking everything that can be changed
delimiter = "\t"
variable_names = ["Frequency", "Intensity1", "Intensity2"]   
Frequency = None 
num_files_per_temp = 20
temperatures = [25, 25, 30, 35, 40, 45, 50, 45, 40, 35, 30, 25, 20, 15, 10, 10]

train_data=[]

temperatures = temperatures[1:]

# Process each group of 20 files
for i in range(num_files_per_temp, len(cycle1_files), num_files_per_temp):
    files_group = cycle1_files[i:i+num_files_per_temp]
    temp = temperatures[(i//num_files_per_temp)-1]   # Get the corresponding temperature for this group
    T = tf.constant(temp, dtype=tf.float64)
# Process each group of 20 files
    ratios = np.array([])

   
    for file in files_group:
        data = pd.read_csv(os.path.join(folder_cycle1, file), delimiter=delimiter, header=None, names=variable_names)

        ratio = np.divide(data['Intensity2'], data['Intensity1'])
        if ratios.size == 0:
            ratios = np.array([ratio])
        else:
            ratios = np.vstack((ratios, [ratio]))  # Add ratio to the numpy array

    avg_intensity = np.mean(ratios, axis=0)
    if Frequency is None:
        Frequency = data['Frequency'].values
        # Assuming Frequency is in Hz
        Frequency_GHz = Frequency / 1e9
        start_frequency = np.min(Frequency)/1e9

    end_frequency = np.max(Frequency)/1e9

    N = Frequency.shape[0]
    dt = np.round((end_frequency - start_frequency) / N, 4)

    timespace = np.linspace(start_frequency, end_frequency, num=N)
    sim_val = _get_vals(T, P)
    noise_sample= avg_intensity[np.where(np.abs(timespace)<2.85)[0]] 
    noise_mean = np.mean(noise_sample)
    avg_intensity = avg_intensity - noise_mean
    avg_intensity = np.max(sim_val)*( avg_intensity)/(np.max(avg_intensity))
    noise_sample= avg_intensity[np.where(np.abs(timespace)<2.85)[0]]
    std_noise=np.std(noise_sample)
    train_data.append(avg_intensity)


all_data_train_2D = np.array(train_data)



pca = PCA(n_components=3)  # Use 3 principal components
pca_train_data = pca.fit_transform(all_data_train_2D) #project the data onto the new lower-dimensional space 

#calculates the principal components (modes) from the training data. This involves computing the mean of the data, centering the data, and then finding the directions (principal components) that maximize the variance in the data.

# Perform PCA on the testing data
pca_test_data = pca.transform(all_data_test_2D) # applies the same transformation learned from the training data to the test data, ensuring consistency in how the dimensionality reduction is applied across your dataset

# Map the principal components to temperatures using linear regression
reg = LinearRegression().fit(pca_train_data, temperatures) # finding coefficients for the linear regression model
# cycle1: Train
# Predict the temperatures for the testing data
predicted_temperatures = reg.predict(pca_test_data)

# Calculate the RMSE between the predicted and actual temperatures
rmse = root_mean_squared_error(temperatures_cycle2, predicted_temperatures)

print('Testing RMSE:', rmse) # 1.392981806830841

predicted_train_temperatures = reg.predict(pca_train_data)
train_rmse = root_mean_squared_error(temperatures, predicted_train_temperatures)

print('Training RMSE:', train_rmse) # 1.6101599788982255
# (tau, adev, _, _) = allantools.adev(predicted_temperatures)

# # Find the minimum Allan deviation (sensitivity)
# sensitivity = min(adev)

# print(f"Sensitivity (minimum Allan deviation): {sensitivity}")



# Plot Modes

# Assuming that pca.components_ is your principal components matrix where each row is a principal component
# Transpose the matrix to get each column as a principal component


components = np.transpose(pca.components_)

# Create a figure
plt.figure(figsize=(6, 4))

# Plot the first principal component
plt.plot(timespace, components[:, 1], label='Mode 3')

plt.xlabel('Frequency (GHz)', fontsize=12)
plt.ylabel('Intensity (arb. units)', fontsize=12)
plt.legend(fontsize=12)  # Add legend to the plot
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
# plt.savefig('D:\\Downloads\\ODMR Paper Figs\\mode_3_cycle2_.png', dpi=300) 
plt.show()
# Create a figure with 3 subplots
fig, axs = plt.subplots(3, figsize=(6, 6))

# Plot each principal component on a separate subplot
for i in range(3):
    axs[i].plot(timespace, components[:, i], label='Mode ' + str(i+1))
    axs[i].set_title('Mode ' + str(i+1))
    axs[i].set_xlabel('Frequency (GHz)', fontsize=12)
    axs[i].set_ylabel('Intensity (arb. units)',fontsize=12)

# Adjust the layout
plt.tight_layout()

# Save the figure
#plt.savefig('D:\\Downloads\\ODMR Paper Figs\\modes_cycle2_normalization.png', dpi=300) 

# Show the plot
plt.show()

# # # Create a figure
# plt.figure(figsize=(10, 7))

# # Plot each principal component
# for i in range(3):
#     plt.plot(components[:, i], label='Mode ' + str(i+1))

# # Set the title and labels

# plt.xlabel('Frequency (GHz)')
# plt.ylabel('Intensity')

# # Add a legend
# plt.legend()

# # Show the plot
# plt.show()


# # Testing
coefficients = np.polyfit(temperatures, predicted_temperatures, 1)
polynomial = np.poly1d(coefficients)
slope = coefficients[0]

# Generate x values
x_values = np.linspace(min(temperatures), max(temperatures), 100)

# Generate y values
y_values = polynomial(x_values)

true_poly= polynomial(temperatures)
# Your predicted values
predicted =predicted_temperatures

r_squared_poly = r2_score(true_poly, predicted)
# Calculate RMSE
rmse_poly = root_mean_squared_error(true_poly, predicted)
# Plot the data points
plt.plot(temperatures, predicted_temperatures, 'o')
plt.plot(x_values, y_values, '-')
plt.xlabel('True Temperatures')
plt.ylabel('Estimated Temperatures')
plt.title('True vs Estimated for Sensor I')
plt.text(min(temperatures), max(predicted) * 0.9, 'R-squared = {:.3f}\nSlope = {:.2f}\nRMSE = {:.2f}'.format(r_squared_poly, slope, rmse_poly), color='red', bbox=dict(facecolor='white', alpha=0.7))
#plt.text(min(temperatures), max(predicted) * 0.9, 'R-squared = {:.2f}\nSlope = {:.2f}\nRMSE = {:.2f}'.format(r_squared_poly, slope, rmse_poly), color='red', bbox=dict(facecolor='white', alpha=0.7))
plt.show()

# # # For Training
coefficients = np.polyfit(temperatures_cycle2,predicted_train_temperatures, 1)
polynomial = np.poly1d(coefficients)
slope = coefficients[0]

# Generate x values
x_values = np.linspace(min(temperatures_cycle2), max(temperatures_cycle2), 100)

# Generate y values
y_values = polynomial(x_values)

true_poly= polynomial(temperatures_cycle2)
# Your predicted values
predicted =predicted_train_temperatures
r_squared_poly = r2_score(true_poly, predicted)
# Calculate RMSE
rmse_poly = root_mean_squared_error(true_poly, predicted)
# Plot the data points
plt.plot(temperatures_cycle2, predicted_train_temperatures, 'o')
plt.plot(x_values, y_values, '-')
plt.xlabel('True Temperatures')
plt.ylabel('Estimated Temperatures')
plt.title('True vs Estimated for Sensor I')
plt.text(min(temperatures_cycle2), max(predicted) * 0.9, 'R-squared = {:.3f}\nSlope = {:.2f}\nRMSE = {:.2f}'.format(r_squared_poly, slope, rmse_poly), color='red', bbox=dict(facecolor='white', alpha=0.7))
#plt.text(min(temperatures), max(predicted) * 0.9, 'R-squared = {:.2f}\nSlope = {:.2f}\nRMSE = {:.2f}'.format(r_squared_poly, slope, rmse_poly), color='red', bbox=dict(facecolor='white', alpha=0.7))
plt.show()



# # # QUADRATIC FIT for training
coefficients = np.polyfit(temperatures_cycle2,  predicted_train_temperatures, 2)
polynomial = np.poly1d(coefficients)
first_coefficient = coefficients[0]

# Generate x values
x_values = np.linspace(min(temperatures_cycle2), max(temperatures_cycle2), 100)

# Generate y values
y_values = polynomial(x_values)

true_poly= polynomial(temperatures_cycle2)
# Your predicted values
predicted =predicted_train_temperatures

r_squared_poly = r2_score(true_poly, predicted)
# Calculate RMSE
rmse_poly = root_mean_squared_error(true_poly, predicted)
# Plot the data points
plt.plot(temperatures_cycle2, predicted_train_temperatures, 'o')
plt.plot(x_values, y_values, '-')
plt.xlabel('True Temperatures')
plt.ylabel('Estimated Temperatures')
plt.title('True vs Estimated for Sensor I ')
plt.text(min(temperatures_cycle2), max(predicted) * 0.9, 'R-squared = {:.3f}\nFirst Coefficient = {:.2e}\nRMSE = {:.2f}'.format(r_squared_poly, first_coefficient, rmse_poly), color='red', bbox=dict(facecolor='white', alpha=0.7))
plt.show()

# # # Quadratic Fit for testing
coefficients = np.polyfit(temperatures, predicted_temperatures, 2)
polynomial = np.poly1d(coefficients)
first_coefficient = coefficients[0]

# Generate x values
x_values = np.linspace(min(temperatures), max(temperatures), 100)

# Generate y values
y_values = polynomial(x_values)

true_poly= polynomial(temperatures)
# Your predicted values
predicted =predicted_temperatures

r_squared_poly = r2_score(true_poly, predicted)
# Calculate RMSE
rmse_poly = root_mean_squared_error(true_poly, predicted)
# Plot the data points
plt.plot(temperatures, predicted_temperatures, 'o')
plt.plot(x_values, y_values, '-')
plt.xlabel('True Temperatures')
plt.ylabel('Estimated Temperatures')
plt.title('True vs Estimated for Sensor I ')
plt.text(min(temperatures), max(predicted) * 0.9, 'R-squared = {:.3f}\nFirst Coefficient = {:.2e}\nRMSE = {:.2f}'.format(r_squared_poly, first_coefficient, rmse_poly), color='red', bbox=dict(facecolor='white', alpha=0.7))
plt.show()


# Combined Figure (Testing)
plt.figure(figsize=(6, 4))

# Fit a linear polynomial
coefficients1 = np.polyfit(temperatures, predicted_temperatures, 1)
polynomial1 = np.poly1d(coefficients1)
slope = coefficients1[0]
x_values = np.linspace(min(temperatures), max(temperatures), 100)
y_values1 = polynomial1(x_values)
true_poly1 = polynomial1(temperatures)
predicted =predicted_temperatures
r_squared_poly1 = r2_score(true_poly1, predicted)

# Plot the linear fit in red
plt.plot(temperatures, predicted_temperatures, 'o', color='green')  # Change color to green
plt.plot(x_values, y_values1, '-', color='red')

# Fit a quadratic polynomial
coefficients2 = np.polyfit(temperatures, predicted_temperatures, 2)
polynomial2 = np.poly1d(coefficients2)
coefficient = coefficients2[0]
y_values2 = polynomial2(x_values)
true_poly2 = polynomial2(temperatures)
predicted =predicted_temperatures
r_squared_poly2 = r2_score(true_poly2, predicted)

# Plot the quadratic fit in blue with dashed line
plt.plot(x_values, y_values2, '--', color='blue')

# Set the title and labels
plt.xlabel('Measured Temperatures (℃)', fontsize=12)
plt.ylabel('Predicted Temperatures (℃)', fontsize=12)

# Add a legend
plt.legend(['Data', 'Linear Fit', 'Quadratic Fit'])

# Add text boxes for the linear and quadratic fits
plt.text(min(temperatures), max(predicted) * 0.5, 'Linear Fit:\nR-squared = {:.3f}\nSlope = {:.2f}'.format(r_squared_poly1, slope), color='red', bbox=dict(facecolor='white', alpha=0.7))
plt.text(max(temperatures)*0.489, max(predicted)*0.3, 'Quadratic Fit:\nR-squared = {:.3f}\n1st Coefficient = {:.2e}'.format(r_squared_poly2, coefficient), color='blue', bbox=dict(facecolor='white', alpha=0.7))

# Adjust the layout to prevent overlap
plt.tight_layout()

# Save the figure
plt.savefig('D:\\Downloads\\ODMR Paper Figs\\combined_fit_testing_2_1.png', dpi=300) 

# Show the plot
plt.show()

# Combined Figure (Training)


plt.figure(figsize=(6, 4))

# Fit a linear polynomial
coefficients1 = np.polyfit(temperatures_cycle2, predicted_train_temperatures, 1)
polynomial1 = np.poly1d(coefficients1)
slope = coefficients1[0]
x_values = np.linspace(min(temperatures_cycle2), max(temperatures_cycle2), 100)
y_values1 = polynomial1(x_values)
true_poly1 = polynomial1(temperatures_cycle2)
predicted =predicted_train_temperatures
r_squared_poly1 = r2_score(true_poly1, predicted)

# Plot the linear fit in red
plt.plot(temperatures_cycle2, predicted_train_temperatures, 'o', color='green')  # Change color to green
plt.plot(x_values, y_values1, '-', color='red')

# Fit a quadratic polynomial
coefficients2 = np.polyfit(temperatures_cycle2, predicted_train_temperatures, 2)
polynomial2 = np.poly1d(coefficients2)
coefficient = coefficients2[0]
y_values2 = polynomial2(x_values)
true_poly2 = polynomial2(temperatures_cycle2)
predicted =predicted_train_temperatures
r_squared_poly2 = r2_score(true_poly2, predicted)

# Plot the quadratic fit in blue with dashed line
plt.plot(x_values, y_values2, '--', color='blue')

# Set the title and labels
plt.xlabel('Measured Temperatures (℃)', fontsize=12)
plt.ylabel('Predicted Temperatures (℃)', fontsize=12)

# Add a legend
plt.legend(['Data', 'Linear Fit', 'Quadratic Fit'])

# Add text boxes for the linear and quadratic fits
plt.text(min(temperatures), max(predicted) * 0.76, 'Linear Fit:\nR-squared = {:.3f}\nSlope = {:.2f}'.format(r_squared_poly1, slope), color='red', bbox=dict(facecolor='white', alpha=0.7))
plt.text(max(temperatures)*0.4, max(predicted)*0.01, 'Quadratic Fit:\nR-squared = {:.3f}\n1st Coefficient = {:.2e}'.format(r_squared_poly2, coefficient), color='blue', bbox=dict(facecolor='white', alpha=0.7))

# Adjust the layout to prevent overlap
plt.tight_layout()

# Save the figure
plt.savefig('D:\\Downloads\\ODMR Paper Figs\\combined_fit_training_2_1.png', dpi=300) 

# Show the plot
plt.show()



# Combined Figure (Testing) in Kelvin

# Convert temperatures and predicted_temperatures from Celsius to Kelvin
temperatures_kelvin = [temp + 273.15 for temp in temperatures]
predicted_temperatures_kelvin = [temp + 273.15 for temp in predicted_temperatures]

plt.figure(figsize=(6, 4))

# Fit a linear polynomial
coefficients1 = np.polyfit(temperatures_kelvin, predicted_temperatures_kelvin, 1)
polynomial1 = np.poly1d(coefficients1)
slope = coefficients1[0]
x_values = np.linspace(min(temperatures_kelvin), max(temperatures_kelvin), 100)
y_values1 = polynomial1(x_values)
true_poly1 = polynomial1(temperatures_kelvin)
predicted = predicted_temperatures_kelvin
r_squared_poly1 = r2_score(true_poly1, predicted)

# Plot the linear fit in red
plt.plot(temperatures_kelvin, predicted_temperatures_kelvin, 'o', color='green')  # Change color to green
plt.plot(x_values, y_values1, '-', color='red')

# # Fit a quadratic polynomial
# coefficients2 = np.polyfit(temperatures_kelvin, predicted_temperatures_kelvin, 2)
# polynomial2 = np.poly1d(coefficients2)
# coefficient = coefficients2[0]
# y_values2 = polynomial2(x_values)
# true_poly2 = polynomial2(temperatures_kelvin)
# predicted = predicted_temperatures_kelvin
# r_squared_poly2 = r2_score(true_poly2, predicted)

# # Plot the quadratic fit in blue with dashed line
# plt.plot(x_values, y_values2, '--', color='blue')

# Set the title and labels
plt.xlabel('Measured Temperatures (K)', fontsize=12)  # Change label to Kelvin
plt.ylabel('Predicted Temperatures (K)', fontsize=12)  # Change label to Kelvin

# Add a legend
plt.legend(['Data', 'Linear Fit'])

plt.annotate('Linear Fit:\nR-squared = {:.3f}\nSlope = {:.2f}'.format(r_squared_poly1, slope), 
             xy=(0.02, 0.6), xycoords='axes fraction', color='red', 
             bbox=dict(facecolor='white', alpha=0.7))

# plt.annotate('Quadratic Fit:\nR-squared = {:.3f}\n1st Coefficient = {:.2e}'.format(r_squared_poly2, coefficient), 
#              xy=(0.6, 0.05), xycoords='axes fraction', color='blue', 
#              bbox=dict(facecolor='white', alpha=0.7))
# Adjust the layout to prevent overlap
plt.tight_layout()

# Save the figure
plt.savefig('D:\\Downloads\\ODMR Paper Figs\\Paper Figs\\cycle2\\PCA\\testing_cycle_1_linear.png', dpi=300) 

# Show the plot
plt.show()

from sklearn.metrics import mean_squared_error

# Calculate RMSE for the linear fit
rmse_poly1 = mean_squared_error(true_poly1, predicted, squared=False)

# Calculate RMSE for the quadratic fit
rmse_poly2 = mean_squared_error(true_poly2, predicted, squared=False)

print(f'RMSE for Linear Fit: {rmse_poly1}') # 1.1346503764818607
print(f'RMSE for Quadratic Fit: {rmse_poly2}') # 1.0821033942256406





# Combined Figure (Training) in Kelvin
# Convert temperatures_cycle2 and predicted_train_temperatures from Celsius to Kelvin
temperatures_cycle2_kelvin = [temp + 273.15 for temp in temperatures_cycle2]
predicted_train_temperatures_kelvin = [temp + 273.15 for temp in predicted_train_temperatures]

plt.figure(figsize=(6, 4))

# Fit a linear polynomial
coefficients1 = np.polyfit(temperatures_cycle2_kelvin, predicted_train_temperatures_kelvin, 1)
polynomial1 = np.poly1d(coefficients1)
slope = coefficients1[0]
x_values = np.linspace(min(temperatures_cycle2_kelvin), max(temperatures_cycle2_kelvin), 100)
y_values1 = polynomial1(x_values)
true_poly1 = polynomial1(temperatures_cycle2_kelvin)
predicted = predicted_train_temperatures_kelvin
r_squared_poly1 = r2_score(true_poly1, predicted)

# Plot the linear fit in red
plt.plot(temperatures_cycle2_kelvin, predicted_train_temperatures_kelvin, 'o', color='green')  # Change color to green
plt.plot(x_values, y_values1, '-', color='red')

# # Fit a quadratic polynomial
# coefficients2 = np.polyfit(temperatures_cycle2_kelvin, predicted_train_temperatures_kelvin, 2)
# polynomial2 = np.poly1d(coefficients2)
# coefficient = coefficients2[0]
# y_values2 = polynomial2(x_values)
# true_poly2 = polynomial2(temperatures_cycle2_kelvin)
# predicted = predicted_train_temperatures_kelvin
# r_squared_poly2 = r2_score(true_poly2, predicted)

# # Plot the quadratic fit in blue with dashed line
# plt.plot(x_values, y_values2, '--', color='blue')

# Set the title and labels
plt.xlabel('Measured Temperatures (K)', fontsize=12)  # Change label to Kelvin
plt.ylabel('Predicted Temperatures (K)', fontsize=12)  # Change label to Kelvin

# Add a legend
plt.legend(['Data', 'Linear Fit'])


plt.annotate('Linear Fit:\nR-squared = {:.3f}\nSlope = {:.2f}'.format(r_squared_poly1, slope), 
             xy=(0.02, 0.6), xycoords='axes fraction', color='red', 
             bbox=dict(facecolor='white', alpha=0.7))

# plt.annotate('Quadratic Fit:\nR-squared = {:.3f}\n1st Coefficient = {:.2e}'.format(r_squared_poly2, coefficient), 
#              xy=(0.6, 0.05), xycoords='axes fraction', color='blue', 
#              bbox=dict(facecolor='white', alpha=0.7))
# Adjust the layout to prevent overlap
plt.tight_layout()

# Save the figure
plt.savefig('D:\\Downloads\\ODMR Paper Figs\\Paper Figs\\cycle2\\PCA\\training_cycle_2_linear.png', dpi=300) 

# Show the plot
plt.show()

# RMSE for Linear Fit: 1.6067132596878735
#RMSE for Quadratic Fit: 1.5932320456585694




# cycle 1 ( avg of intensities and then ratios )
# # num_files_per_temp = 20
# temperatures = [25, 25, 30, 35, 40, 45, 50, 45, 40, 35, 30, 25, 20, 15, 10, 10]

# all_data_test_avg = []
# temperatures = temperatures[1:]
# file_counts = {}

# for i in range(num_files_per_temp, len(cycle1_files), num_files_per_temp):
#     files_group = cycle1_files[i:i+num_files_per_temp]
#     temp = temperatures[(i//num_files_per_temp)-1]   # Get the corresponding temperature for this group
#     T = tf.constant(temp, dtype=tf.float64)
#     intensity1_test = np.array([])
#     intensity2_test = np.array([])


#     for file in files_group:
#         data = pd.read_csv(os.path.join(folder_cycle1, file), delimiter=delimiter, header=None, names=variable_names)
#         if intensity1_test.size == 0:
#             intensity1_test = np.array([data['Intensity1']])
#             intensity2_test = np.array([data['Intensity2']])
#         else:
#             intensity1_test = np.vstack((intensity1_test, [data['Intensity1']]))
#             intensity2_test = np.vstack((intensity2_test, [data['Intensity2']]))

#     avg_intensity1_test = np.mean(intensity1_test, axis=0)
#     avg_intensity2_test = np.mean(intensity2_test, axis=0)
#     avg_intensity= np.divide(avg_intensity2_test, avg_intensity1_test)
    
#     if Frequency is None:
#         Frequency = data['Frequency'].values
#         # Assuming Frequency is in Hz
#         Frequency_GHz = Frequency / 1e9
#         start_frequency = np.min(Frequency)/1e9

#     end_frequency = np.max(Frequency)/1e9

#     N = Frequency.shape[0]
#     dt = np.round((end_frequency - start_frequency) / N, 4)

#     timespace = np.linspace(start_frequency, end_frequency, num=N)
#     sim_val = _get_vals(T, P)
#     noise_sample= avg_intensity[np.where(np.abs(timespace)<2.85)[0]] 
#     noise_mean = np.mean(noise_sample)
#     avg_intensity = avg_intensity - noise_mean
#     avg_intensity = np.max(sim_val)*( avg_intensity)/(np.max(avg_intensity))
#     noise_sample= avg_intensity[np.where(np.abs(timespace)<2.85)[0]]
#     std_noise=np.std(noise_sample)
#     all_data_test_avg.append(avg_intensity)

# all_data_test_2D = np.array(all_data_test_avg)




# # Cycle 2 only ratios 
# train_data=[]


# # Reading Data and taking everything that can be changed
# delimiter = "\t"
# variable_names = ["Frequency", "Intensity1", "Intensity2"]   
# Frequency = None 
# num_files_per_temp = 2
# #temperatures = [25, 25, 30, 35, 40, 45, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5]
# temperatures_cycle2 = [-30.0, -20.0, -10.0,  0.0, 10.0,  20.0, 30.0,  40.0, 50.0,40.0, 30.0,  20.0, 10.0, 0.0, -10.0, -20.0, -30.0]


# # Process each group of 20 files
# for i in range(0, len(cycle2_files), num_files_per_temp):
#     files_group = cycle2_files[i:i+num_files_per_temp]
#     temp = temperatures_cycle2[i//num_files_per_temp]  # Get the corresponding temperature for this group
#     T = tf.constant(temp, dtype=tf.float64)
#     ratios = np.array([])

   
#     for file in files_group:
#         data = pd.read_csv(os.path.join(folder_cycle2, file), delimiter=delimiter, header=None, names=variable_names)

#         avg_intensity_cycle2 = np.divide(data['Intensity2'], data['Intensity1'])

    
#     if Frequency is None:
#         Frequency = data['Frequency'].values
#         # Assuming Frequency is in Hz
#         Frequency_GHz = Frequency / 1e9
#         start_frequency = np.min(Frequency)/1e9

#     end_frequency = np.max(Frequency)/1e9

#     N = Frequency.shape[0]
#     dt = np.round((end_frequency - start_frequency) / N, 4)

#     timespace = np.linspace(start_frequency, end_frequency, num=N)
#     sim_val = _get_vals(T, P)
#     noise_sample= avg_intensity_cycle2[np.where(np.abs(timespace)<2.85)[0]] 
#     noise_mean = np.mean(noise_sample)
#     avg_intensity_cycle2 = avg_intensity_cycle2 - noise_mean
#     avg_intensity_cycle2 = np.max(sim_val)*( avg_intensity_cycle2)/(np.max(avg_intensity_cycle2))
#     noise_sample= avg_intensity_cycle2[np.where(np.abs(timespace)<2.85)[0]]
#     std_noise=np.std(noise_sample)
#     train_data.append(avg_intensity_cycle2)
    




# all_data_train_2D = np.array(train_data)




# Perform PCA
# First Mode for testing
pca = PCA(n_components=3)  # Adjust the number of components as needed
pca.fit(all_data_test_2D)
components = np.transpose(pca.components_)

# Create a figure
plt.figure(figsize=(6, 4))

# Plot the first principal component
plt.plot(timespace, components[:, 0], label='Mode 1')

plt.xlabel('Frequency (GHz)', fontsize=12)
plt.ylabel('Intensity (arb. units)', fontsize=12)
plt.legend(fontsize=12)  # Add legend to the plot
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('D:\\Downloads\\ODMR Paper Figs\\mode_1_testing.png', dpi=300) 
plt.show()

pca = PCA(n_components=3)  # Adjust the number of components as needed
pca.fit(all_data_test_2D)
components = np.transpose(pca.components_)

# Create a figure
plt.figure(figsize=(6, 4))

# Plot the first principal component
plt.plot(timespace, components[:, 1], label='Mode 2')

plt.xlabel('Frequency (GHz)', fontsize=12)
plt.ylabel('Intensity (arb. units)', fontsize=12)
plt.legend(fontsize=12)  # Add legend to the plot
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('D:\\Downloads\\ODMR Paper Figs\\mode_2_testing.png', dpi=300) 
plt.show()

pca = PCA(n_components=3)  # Adjust the number of components as needed
pca.fit(all_data_test_2D)
components = np.transpose(pca.components_)

# Create a figure
plt.figure(figsize=(6, 4))

# Plot the first principal component
plt.plot(timespace, components[:, 2], label='Mode 3')

plt.xlabel('Frequency (GHz)', fontsize=12)
plt.ylabel('Intensity (arb. units)', fontsize=12)
plt.legend(fontsize=12)  # Add legend to the plot
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('D:\\Downloads\\ODMR Paper Figs\\mode_3_testing.png', dpi=300) 
plt.show()



pca = PCA(n_components=3)  # Adjust the number of components as needed
pca.fit(all_data_test_2D)
components = np.transpose(pca.components_)

# Create a figure with 3 subplots
fig, axs = plt.subplots(3, figsize=(14, 7))

# Plot each principal component on a separate subplot
for i in range(3):
    axs[i].plot(timespace, components[:, i], label='Mode ' + str(i+1))
    
    axs[i].set_xlabel('Frequency (GHz)', fontsize=12)
    axs[i].set_ylabel('Intensity (arb. units)', fontsize=12)
    axs[i].legend(fontsize=12)  # Add legend to each subplot
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

#plt.savefig('D:\\Downloads\\ODMR Paper Figs\\cycle1_modes.png', dpi=300)

# Perform PCA
pca = PCA(n_components=3)  # Adjust the number of components as needed
pca.fit(all_data_test_2D)
components = np.transpose(pca.components_)

# Create a figure with 3 subplots
fig, axs = plt.subplots(3, figsize=(7, 7))

# Plot each principal component on a separate subplot
for i in range(3):
    axs[i].plot(timespace, components[:, i], label='Mode ' + str(i+1))
    
    axs[i].set_xlabel('Frequency (GHz)', fontsize=12)
    axs[i].set_ylabel('Intensity (arb. units)', fontsize=12)
    axs[i].legend(fontsize=12)  # Add legend to each subplot
    axs[i].set_aspect('equal')  # Make the subplot square

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()


components = np.transpose(pca.components_)

# Create a figure
plt.figure(figsize=(6, 4))

# Plot the second principal component
plt.plot(timespace, components[:, 2], label='Mode 3')

plt.xlabel('Frequency (GHz)', fontsize=12)
plt.ylabel('Intensity (arb. units)', fontsize=12)
plt.legend(fontsize=12)  # Add legend to the plot
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('D:\\Downloads\\ODMR Paper Figs\\mode_3_training.png', dpi=300) 
plt.show()