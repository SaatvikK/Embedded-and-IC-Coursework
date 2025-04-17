# %%
################ IMPORTS ################
#!pip3 install torch joblib scikit-learn pandas numpy matplotlib seaborn # Ensure all needed libs are installed
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings
import joblib # For saving/loading scaler
import os     # For checking files, CWD (optional but good practice)

warnings.filterwarnings('ignore')
#########################################


# %%
################ Initialising ################
def pcInit():
  # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Use if your machine is Windows or Linux
  device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # Use if your machine is MacOS
  print(f"Using device: {device}") # Debugging to check if the device is set up correctly!
  return device

# MODIFIED: Added filepath_override argument
def openSet(defaultPath = True, filepath_override=None) -> pd.DataFrame | None:
  """Opens the CSV file using pandas' in-built method"""
  if filepath_override:
      filepath = filepath_override
  elif defaultPath:
      # Sensible default if file is expected in the same directory as the script
      script_dir = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
      filepath = os.path.join(script_dir, "london_weather.csv")
  else:
      filepath = input("Enter file location: ")

  print(f"Attempting to load data from: {filepath}") # More explicit print

  try:
      # Check if file actually exists before trying to read
      if not os.path.exists(filepath):
           raise FileNotFoundError(f"File does not exist at the specified path: {filepath}")
      df = pd.read_csv(filepath)
      print(f"Successfully loaded data from: {filepath}")
      return df
  except FileNotFoundError as fnf_error:
      print(f"Error: {fnf_error}")
      print("Please ensure the file exists and the path is correct.")
      return None
  except Exception as e:
        print(f"An error occurred opening the file: {e}")
        return None

device = pcInit()
# --- Important: Make sure openSet can find your file ---
# Use defaultPath=False to trigger the input prompt
# If london_weather.csv is in the same dir as h.py, defaultPath=True might work if run directly
df_initial = openSet(defaultPath=False) # Prompt user for path

# Exit if dataframe loading failed
if df_initial is None:
    raise SystemExit("Could not load dataframe. Exiting.")

# Display first few rows if loaded successfully
print("Initial Data Head:")
print(df_initial.head())

# Store the path provided by the user (or the default) for later use in inference
# This assumes the first call to openSet gets the primary data file path.
# If using defaultPath=True, manually set this variable.
try:
    initial_file_path = filepath # Relies on filepath being set in openSet's scope - might be fragile
    # Safer way: Get it from the input or explicitly define it
    # For now, let's assume the user input during the first openSet call is the one we need later.
    # We'll pass this path explicitly to inference functions.
    # If using defaultPath=True, set initial_file_path manually here:
    # initial_file_path = "london_weather.csv" # Or the full path used
    print(f"Will use this path for loading data later if needed: {initial_file_path}")
except NameError:
     print("Warning: Could not automatically determine initial file path. Set 'initial_file_path' manually if needed.")
     initial_file_path = None # Set manually if needed below

##############################################

# %%
################ Data Pre-Processing #############

# Turn the date string objects into actual datetime objects
def dateTimeFmt(df):
  df = df.copy() # Avoid SettingWithCopyWarning
  df.loc[:, "date"] = pd.to_datetime(df["date"], format='%Y%m%d')
  return df

# MODIFIED: No inplace=True, returns cleaned DataFrame
def naVals(df, mode="interpolate", printNAs=True):
    """Handles missing values in the DataFrame."""
    df_cleaned = df.copy() # Work on a copy

    if printNAs:
        print("\nNumber of N/A observations per feature (before cleaning):")
        print(df_cleaned.isna().sum())

    if mode == "interpolate":
        numeric_cols = df_cleaned.select_dtypes(include=np.number).columns
        # Apply interpolation - returns modified DataFrame
        if not numeric_cols.empty:
             df_cleaned[numeric_cols] = df_cleaned[numeric_cols].interpolate(method="linear", limit_direction='both')
        else:
             print("Warning: No numeric columns found for interpolation.")
    elif mode == "rmv":
        df_cleaned.dropna(inplace=True) # Dropna can stay inplace for removal
    else: # ffill or bfill
         # fillna returns modified DataFrame when inplace=False (default)
        df_cleaned = df_cleaned.fillna(method=mode)

    # Fallback: Fill any remaining NaNs AFTER main method
    # Apply fillna without inplace=True
    df_cleaned = df_cleaned.fillna(method='ffill') # Forward fill
    df_cleaned = df_cleaned.fillna(method='bfill') # Backward fill

    if printNAs:
        print("\nAfter cleaning:")
        print(df_cleaned.isna().sum())
        final_na_count = df_cleaned.isna().sum().sum()
        if final_na_count > 0:
            print(f"Warning: {final_na_count} NaNs still present after cleaning!")

    # --- Make sure to return the cleaned dataframe ---
    return df_cleaned


# Scales/Normalises Data using a provided scaler or fits a new one
def dataScale(df_numeric, scaler=None, fit_scaler=False):
    """
    Scales the numeric data.
    If fit_scaler is True, fits a new StandardScaler.
    If scaler is provided, uses it to transform the data.
    Returns the scaled data and the scaler used.
    """
    if scaler is None:
        if fit_scaler:
            print("\nFitting a new StandardScaler...")
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df_numeric)
            print("Fitted and transformed data with a new scaler.")
        else:
            raise ValueError("Scaler must be provided if fit_scaler is False")
    else:
        # print("Transforming data with the provided scaler.") # Optional print
        scaled_data = scaler.transform(df_numeric)
    return scaled_data, scaler


# Splits the data into training and testing sets
def dataSplit(df_scaled, original_target_col, T=20, trainPortion=0.8):
    """
    Splits scaled data into sequences for LSTM training/testing.
    Args:
        df_scaled (np.array): The scaled numerical data (output from dataScale).
        original_target_col (pd.Series): The original, unscaled target column values.
        T (int): Timestep sequence length.
        trainPortion (float): Proportion of data for training.

    Returns:
        dict: Dictionary containing datasets and dimensions.
    """
    N = len(df_scaled) - T  # Total number of sequences we can make
    if N < 1:
         raise ValueError(f"Dataset too small ({len(df_scaled)} rows) to create sequences of length T={T}")

    trainingSize = int(N * trainPortion) # Number of training sequences
    testSize = N - trainingSize      # Number of test sequences
    if trainingSize <= 0 or testSize <=0:
        raise ValueError(f"Training portion ({trainPortion}) results in 0 sequences for train ({trainingSize}) or test ({testSize}). Adjust portion or check data size ({N} total sequences).")

    setSize = df_scaled.shape[1]     # Number of features

    # Ensure we don't try to access indices beyond the original target length
    max_target_idx = len(original_target_col) - 1

    print(f"\nCreating sequences (T={T})...")
    # Train:
    Xtrain = np.zeros((trainingSize, T, setSize))
    Ytrain = np.zeros((trainingSize, 1))
    actual_train_count = 0
    for t in range(trainingSize):
        seq_end_idx = t + T
        target_idx = seq_end_idx # Target is the value AFTER the sequence end
        if target_idx > max_target_idx:
            print(f"Warning: Reached end of target data during training sequence creation at index {t}. Max target index: {max_target_idx}")
            break # Prevent index out of bounds for Ytrain
        Xtrain[t, :, :] = df_scaled[t:seq_end_idx]
        Ytrain[t] = original_target_col.iloc[target_idx]
        actual_train_count += 1

    # Adjust training arrays if we broke early
    if actual_train_count < trainingSize:
         print(f"Adjusting training size from {trainingSize} to {actual_train_count} due to data limits.")
         Xtrain = Xtrain[:actual_train_count]
         Ytrain = Ytrain[:actual_train_count]
         trainingSize = actual_train_count # Update size

    # Test
    Xtest = np.zeros((testSize, T, setSize))
    Ytest = np.zeros((testSize, 1))
    actual_test_count = 0
    for i in range(testSize):
        t = i + trainingSize # Overall index in the scaled data for start of sequence
        seq_end_idx = t + T
        target_idx = seq_end_idx
        if target_idx > max_target_idx:
             print(f"Warning: Reached end of target data during test sequence creation at index {i}. Max target index: {max_target_idx}")
             break # Prevent index out of bounds for Ytest
        Xtest[i, :, :] = df_scaled[t:seq_end_idx]
        Ytest[i] = original_target_col.iloc[target_idx]
        actual_test_count +=1

    # Resize test arrays if we broke early
    if actual_test_count < testSize:
        print(f"Adjusting test size from {testSize} to {actual_test_count} due to data limits.")
        Xtest = Xtest[:actual_test_count]
        Ytest = Ytest[:actual_test_count]
        testSize = actual_test_count # Update size

    print(f"Created {trainingSize} training sequences and {testSize} testing sequences.")

    XTrain = torch.from_numpy(Xtrain.astype(np.float32))
    YTrain = torch.from_numpy(Ytrain.astype(np.float32))
    XTest = torch.from_numpy(Xtest.astype(np.float32))
    YTest = torch.from_numpy(Ytest.astype(np.float32))

    return {
      "datasets": [XTrain, YTrain, XTest, YTest],
      "dimensions": {
          "timesteps": T,
          "dimension": setSize,
          "N_total_sequences": N,
          "N_train": trainingSize,
          "N_test": testSize
          }
      }
#####################################################

# --- Main Preprocessing Function for Training ---
def preprocess_train_test(df: pd.DataFrame, trainPercent = 0.8, timestep = 20, scaler_path="scaler.joblib", target_col_name='mean_temp'):
    """
    Preprocesses the entire dataset for training/testing, splits, scales, and saves the scaler.
    """
    print("\n--- Starting Preprocessing for Training/Testing ---")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    # 1. Format Date
    df = dateTimeFmt(df)
    df_dated = df.copy() # Keep a copy with dates for plotting later

    # 2. Handle NaNs
    # Clean NaNs on the dataframe before separating features/target
    df = naVals(df, printNAs=True, mode="interpolate") # Use interpolation + ffill/bfill

    # 3. Separate Target and Features
    if target_col_name not in df.columns:
      raise ValueError(f"Target column '{target_col_name}' not found in dataframe.")
    targets = df[target_col_name].copy() # Unscaled target series

    # Assume features are all columns except 'date' (and potentially target if not used as feature)
    # Make sure 'date' column is handled correctly
    feature_columns = df.drop(columns=['date', target_col_name], errors='ignore').columns.tolist()
    # If target IS also a feature (common in time series):
    if target_col_name in df.columns and target_col_name not in feature_columns:
         feature_columns.append(target_col_name) # Ensure target is included if it's a feature

    # Reorder df to have features in the desired order before scaling
    df_features = df[feature_columns].copy()
    print(f"\nUsing features: {feature_columns}")
    D_features = len(feature_columns) # Dimension of feature space

    # --- 4. Scale the Feature Data ---
    # Fit scaler ONLY on the training portion conceptually. For simplicity here, fitting on all feature data.
    # For strict approach: split df_features first, fit scaler only on train part, then transform both.
    df_scaled, scaler = dataScale(df_features, fit_scaler=True) # Fit a new scaler here
    print(f"Scaler fitted. Mean sample: {scaler.mean_[:3]}..., Scale sample: {scaler.scale_[:3]}...")

    # --- 5. Save the Scaler ---
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # --- 6. Split into Sequences ---
    vars_dict = dataSplit(df_scaled, targets, T=timestep, trainPortion=trainPercent)

    # --- 7. Add supplementary info ---
    vars_dict["dfDated"] = df_dated # Add dated df back for potential plotting
    vars_dict["scaler"] = scaler # Add scaler to the dict
    vars_dict["scaler_path"] = scaler_path # Store path for reference
    vars_dict["feature_names"] = feature_columns # Store feature names in order used
    vars_dict["target_column"] = target_col_name

    print("--- Preprocessing Complete ---")
    return vars_dict

# --- Run Preprocessing for Training ---
T_steps = 20 # Timesteps / Sequence Length
scaler_filename = "london_weather_scaler.joblib" # Define scaler filename
target_column = 'mean_temp'

vars_processed = preprocess_train_test(df_initial,
                                       trainPercent=0.8,
                                       timestep=T_steps,
                                       scaler_path=scaler_filename,
                                       target_col_name=target_column)

XTrain, YTrain, XTest, YTest = vars_processed["datasets"]
T, D = vars_processed["dimensions"]["timesteps"], vars_processed["dimensions"]["dimension"]
N_train, N_test = vars_processed["dimensions"]["N_train"], vars_processed["dimensions"]["N_test"]
dfDated = vars_processed["dfDated"]
scaler_path_used = vars_processed["scaler_path"]
feature_names_used = vars_processed["feature_names"]

print(f"\nData Shapes - Train X: {XTrain.shape}, Y: {YTrain.shape}")
print(f"Data Shapes - Test X: {XTest.shape}, Y: {YTest.shape}")
print(f"Input Dimension (D - features): {D}, Timesteps (T): {T}")
print(f"Feature names used: {feature_names_used}")
print(f"Scaler saved at: {scaler_path_used}")

# %%
#################### Model ####################

class lstmAlgorithm(nn.Module):
  def __init__(self, inputD, hiddenD, layerD, outputD):
    super(lstmAlgorithm, self).__init__()
    self.M = hiddenD # Hidden dimension size
    self.L = layerD # Number of LSTM layers

    # input_size = D (number of features)
    self.rnn = nn.LSTM(input_size=inputD, hidden_size=hiddenD,
                       num_layers=layerD, batch_first=True, dropout=0.2 if layerD > 1 else 0)
    # batch_first=True means input shape is (batch_size, seq_len, features)
    # Added dropout for regularization if using multiple layers

    self.fc = nn.Linear(hiddenD, outputD) # Fully connected layer maps hidden state to output

  def forward(self, X):
    # initial hidden state and cell state
    # Shape: (num_layers, batch_size, hidden_size)
    h0 = torch.zeros(self.L, X.size(0), self.M).to(X.device) # Use X.device to ensure compatibility
    c0 = torch.zeros(self.L, X.size(0), self.M).to(X.device) # Use X.device

    # LSTM returns: output (all hidden states), (hn (final hidden state), cn (final cell state))
    # output shape: (batch_size, seq_len, hidden_size) if batch_first=True
    out, _ = self.rnn(X, (h0, c0)) # We don't need hn, cn explicitly here if using last output step

    # We want the output from the *last* time step for prediction
    # out[:, -1, :] selects the last time step's hidden state for all batches
    # Shape: (batch_size, hidden_size)
    out = self.fc(out[:, -1, :])
    return out

  # --- Training Method ---
  def train_model(self, learningRate, XTrain, YTrain, XTest, YTest, device, epochs = 200, batch_size=64):
      """Trains the LSTM model."""
      self.to(device) # Ensure model is on the correct device

      # Loss and optimiser
      criterion = nn.MSELoss()
      # Consider AdamW for potentially better performance with weight decay
      optimizer = torch.optim.AdamW(self.parameters(), lr=learningRate, weight_decay=1e-4)
      # optimizer = torch.optim.SGD(self.parameters(), lr=learningRate, momentum=0.9, weight_decay=1e-4) # Alternative

      # DataLoaders for batching
      train_dataset = torch.utils.data.TensorDataset(XTrain, YTrain)
      train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

      test_dataset = torch.utils.data.TensorDataset(XTest, YTest)
      test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

      trainLosses = []
      testLosses = []
      print(f"\n--- Starting Training (Epochs: {epochs}, Batch Size: {batch_size}, LR: {learningRate}) ---")

      for epoch in range(epochs):
          self.train() # Set model to training mode (enables dropout etc.)
          batch_train_losses = []
          for i, (X_batch, y_batch) in enumerate(train_loader):
              X_batch, y_batch = X_batch.to(device), y_batch.to(device) # Move batch to device

              optimizer.zero_grad()   # Zero gradients before calculation
              outputs = self(X_batch) # Forward pass
              loss = criterion(outputs, y_batch) # Calculate loss
              loss.backward()         # Backpropagation
              optimizer.step()          # Update weights
              batch_train_losses.append(loss.item())

          epoch_train_loss = np.mean(batch_train_losses)
          trainLosses.append(epoch_train_loss)

          # --- Evaluation phase ---
          self.eval() # Set model to evaluation mode (disables dropout etc.)
          batch_test_losses = []
          with torch.no_grad(): # Disable gradient calculation for evaluation
              for X_batch, y_batch in test_loader:
                  X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                  testOutputs = self(X_batch)
                  testLoss = criterion(testOutputs, y_batch)
                  batch_test_losses.append(testLoss.item())

          epoch_test_loss = np.mean(batch_test_losses)
          testLosses.append(epoch_test_loss)

          # Print progress periodically
          if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
              print(f'Epoch [{epoch+1:03d}/{epochs}], Train Loss: {epoch_train_loss:.4f}, Test Loss: {epoch_test_loss:.4f}')

      print("--- Training Complete ---")
      return trainLosses, testLosses

# %%
# --- Initialize and Train Model ---
# Hyperparameters
hidden_dim = 128 # Reduced hidden dim - 512 might be too large unless data is very complex
num_layers = 2
output_dim = 1   # Predicting one value (mean_temp)
learning_rate = 0.001
num_epochs = 2 # Adjust as needed based on loss plot convergence
batch_proc_size = 64

# Ensure D matches the number of features used
assert D == len(feature_names_used), f"Mismatch: Model input D ({D}) != number of features ({len(feature_names_used)})"

lstmModel = lstmAlgorithm(inputD=D, hiddenD=hidden_dim, layerD=num_layers, outputD=output_dim)

trainLosses, testLosses = lstmModel.train_model(learning_rate, XTrain, YTrain, XTest, YTest,
                                                device=device, epochs=num_epochs, batch_size=batch_proc_size)

# %%
# --- Save Trained Model Weights ---
model_weights_save_path = "lstm_model_weights.pt"
torch.save(lstmModel.state_dict(), model_weights_save_path) # Save only weights (recommended)
print(f"\nModel weights saved to {model_weights_save_path}")

# %%
################ Model Training Evaluation Plot ################
plt.figure(figsize=(10, 6))
plt.plot(trainLosses, label='Train Loss')
plt.plot(testLosses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Test Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.ylim(bottom=0) # Start y-axis at 0 for loss plots
plt.tight_layout()
plt.show()

# %%
################ Model TESTING (One-step ahead on Test Set) ################
print("\n--- Evaluating Model on Test Set (One-Step Ahead) ---")
lstmModel.eval() # Ensure model is in evaluation mode

testPreds_list = []
testTarget_list = []

# Use DataLoader for consistency, or iterate directly if memory allows
test_loader_oneshot = torch.utils.data.DataLoader(
    dataset=torch.utils.data.TensorDataset(XTest, YTest),
    batch_size=batch_proc_size, # Use same batch size or larger for eval
    shuffle=False)

with torch.no_grad():
    for x_batch, y_batch in test_loader_oneshot:
        x_batch = x_batch.to(device)
        outputs = lstmModel(x_batch)
        testPreds_list.extend(outputs.cpu().numpy().flatten()) # Move preds to CPU and flatten
        testTarget_list.extend(y_batch.numpy().flatten())      # Targets are already on CPU if dataloader is standard

testPreds = np.array(testPreds_list)
testTarget = np.array(testTarget_list)

# --- Plotting Test Results ---
# Find the dates corresponding to the YTest values
num_train_sequences = vars_processed["dimensions"]["N_train"]
total_original_rows = len(dfDated) # Original df length

# The first Y_test value corresponds to the target T steps AFTER the start of the first test sequence.
# The first test sequence input starts at index `num_train_sequences` in the scaled data.
# The target for this sequence is at index `num_train_sequences + T` in the original dataframe.
start_index_for_test_targets = num_train_sequences + T

# Calculate end index, ensuring it doesn't exceed original data length
end_index_for_test_targets = start_index_for_test_targets + len(testTarget)
if end_index_for_test_targets > total_original_rows:
    print(f"Warning: Calculated target end index ({end_index_for_test_targets}) exceeds original dataframe length ({total_original_rows}). Adjusting plot range.")
    # Adjust the number of predictions/targets to plot if necessary
    num_targets_to_plot = total_original_rows - start_index_for_test_targets
    testTarget = testTarget[:num_targets_to_plot]
    testPreds = testPreds[:num_targets_to_plot]
    end_index_for_test_targets = total_original_rows # Adjust end index


print(f"Plotting test targets from original df index {start_index_for_test_targets} to {end_index_for_test_targets-1}")

# Select the corresponding dates and actual mean_temp values from the dated dataframe
plot_df_test = dfDated.iloc[start_index_for_test_targets:end_index_for_test_targets][['date', target_column]].copy()

# Ensure the lengths match before assigning predictions
if len(plot_df_test) != len(testPreds):
     print(f"ERROR: Length mismatch! Plotting DF has {len(plot_df_test)} rows, but found {len(testPreds)} predictions.")
     # Attempt to align - this might indicate an off-by-one or other issue in indexing logic
     min_len = min(len(plot_df_test), len(testPreds))
     print(f"Plotting only the first {min_len} available points due to mismatch.")
     plot_df_test = plot_df_test.iloc[:min_len]
     testPreds = testPreds[:min_len] # Slice predictions too

plot_df_test['prediction'] = testPreds
plot_df_test.set_index('date', inplace=True)

# Plot
plt.figure(figsize=(15.0, 7.0))
plt.plot(plot_df_test.index, plot_df_test[target_column], label='Actual Temperature', linewidth=1.5)
plt.plot(plot_df_test.index, plot_df_test['prediction'], label='One-step Prediction (Test Set)', linewidth=1.5, linestyle='--', alpha=0.8)
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title(f'Model Performance on Test Set (Target: {target_column})')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate and print Test Set MSE
test_mse = np.mean((testPreds - testTarget)**2)
print(f"\nTest Set MSE: {test_mse:.4f}")
test_mae = np.mean(np.abs(testPreds - testTarget))
print(f"Test Set MAE: {test_mae:.4f}")


# %%
#################### INFERENCING (Multi-Step Ahead) ####################

# --- Function to prepare data slice for first inference step ---
# MODIFIED: Uses loaded scaler, corrected history slicing & cleaning order
def compile_inference_data(start_date_str, num_days_history: int, full_df_path:str, scaler_path: str, feature_names_expected: list):
    """
    Prepares the input data sequence needed for the *first* inference prediction.
    Args:
        start_date_str (str): The date for which we want the *first* prediction (YYYYMMDD format).
        num_days_history (int): The number of past days needed for one sequence (T).
        full_df_path (str): Path to the original, complete CSV file.
        scaler_path (str): Path to the saved scaler file.
        feature_names_expected (list): List of feature names in the order the model expects.

    Returns:
        torch.Tensor: The input tensor for the first prediction (shape [1, T, D]).
        StandardScaler: The loaded scaler object.
        pd.Timestamp: The actual start date timestamp object.
        pd.DataFrame: The full, cleaned dataframe with date index (for reference/plotting).
    """
    print(f"\n--- Compiling Inference Data ---")
    print(f"Targeting first prediction for date: {start_date_str}")
    print(f"Required history: {num_days_history} days")

    # 1. Load Full Data and Scaler
    # Use override path in openSet
    df_full = openSet(defaultPath=False, filepath_override=full_df_path)
    if df_full is None:
        raise ValueError(f"Failed to load full dataframe for inference from: {full_df_path}")

    try:
        scaler = joblib.load(scaler_path)
        print(f"Scaler loaded successfully from {scaler_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}. Train the model first.")
    except Exception as e:
         raise RuntimeError(f"Error loading scaler: {e}")

    # 2. Preprocess Full Data (Date format and Clean NaNs BEFORE slicing)
    df_full = dateTimeFmt(df_full)

    # --- Clean NaNs on the entire dataframe ---
    print("Cleaning NaNs on the fully loaded dataframe for inference...")
    df_full_cleaned = naVals(df_full, printNAs=False, mode="interpolate") # Use the modified naVals

    # --- Check NaNs in the whole df AFTER cleaning (important debug) ---
    if df_full_cleaned.isna().sum().sum() > 0:
        print("WARNING: NaNs remain in the full dataframe AFTER cleaning in compile_inference_data:")
        print(df_full_cleaned.isna().sum())
        # Decide how to handle this: raise error or try to proceed?
        # raise ValueError("NaNs persist in full dataframe after cleaning attempt.")

    # 3. Find Indices and Set Index on the CLEANED dataframe
    try:
        start_date = pd.to_datetime(start_date_str, format='%Y%m%d')
    except ValueError:
        raise ValueError(f"Invalid start_date_str format: '{start_date_str}'. Expected YYYYMMDD.")

    end_hist_date = start_date - pd.Timedelta(days=1)
    start_hist_date = end_hist_date - pd.Timedelta(days=num_days_history - 1)

    # --- Set index on the CLEANED dataframe ---
    df_full_cleaned = df_full_cleaned.set_index('date')
    df_full_cleaned.sort_index(inplace=True) # Ensure data is sorted by date

    print(f"Required history range (inclusive): {start_hist_date.strftime('%Y-%m-%d')} to {end_hist_date.strftime('%Y-%m-%d')}")

    # 4. Slice the CLEANED DataFrame
    try:
        # Ensure the slice covers the exact range needed
        hist_slice_df = df_full_cleaned.loc[start_hist_date:end_hist_date]
    except KeyError as e:
        print(f"\nError slicing data: Date range likely not found.")
        print(f"Requested Start: {start_hist_date}, Requested End: {end_hist_date}")
        avail_start = df_full_cleaned.index.min()
        avail_end = df_full_cleaned.index.max()
        print(f"Available date range in cleaned data: {avail_start} to {avail_end}")
        if start_hist_date < avail_start or end_hist_date > avail_end:
             print("Reason: Requested history range falls outside the available data dates.")
        else:
             print("Reason: Specific dates within the range might be missing (check for gaps).")
        raise e

    # Verify the number of rows fetched
    if len(hist_slice_df) != num_days_history:
        # This can happen if there are missing dates within the range
        print(f"Warning: Fetched {len(hist_slice_df)} rows for history, but expected {num_days_history}.")
        print("This might be due to missing dates in the source data within the required range.")
        # Option 1: Raise error
        # raise ValueError(f"Insufficient historical data points found ({len(hist_slice_df)}) for the required range. Check for missing dates.")
        # Option 2: Try to proceed if close enough (use with caution)
        if len(hist_slice_df) < num_days_history * 0.9: # Allow small number of missing days?
             raise ValueError(f"Significantly insufficient historical data points found ({len(hist_slice_df)}). Cannot proceed.")
        print("Proceeding, but prediction accuracy may be affected.")
        # If proceeding, ensure the shape matches T later (padding might be needed, but complex)
        # Simplest for now is to error out if exact T days are not found.
        raise ValueError(f"Exact number of history days ({num_days_history}) not found in slice ({len(hist_slice_df)}). Cannot create input sequence.")

    # 5. Select and Order Features FROM THE CLEANED SLICE
    # Ensure feature_names_expected matches the order used during training
    try:
        hist_slice_numeric = hist_slice_df[feature_names_expected].copy()
    except KeyError as e:
        missing_cols = set(feature_names_expected) - set(hist_slice_df.columns)
        print(f"Error: Expected feature columns not found in the historical slice: {missing_cols}")
        raise e

    # --- Final Check for NaNs *in the slice* before scaling (should pass now) ---
    if hist_slice_numeric.isna().sum().sum() > 0:
        print("ERROR: NaNs detected in the final historical slice just BEFORE scaling:")
        print(hist_slice_numeric.isna().sum())
        raise ValueError("NaNs detected in historical slice AFTER cleaning full df and slicing.")

    # 6. Scale the History Slice
    scaled_hist_slice, _ = dataScale(hist_slice_numeric, scaler=scaler, fit_scaler=False) # Use loaded scaler

    # 7. Convert to Tensor
    X_infer_np = scaled_hist_slice.astype(np.float32)
    # Ensure shape is [T, D]
    if X_infer_np.shape != (num_days_history, len(feature_names_expected)):
         raise ValueError(f"Numpy array shape mismatch before tensor conversion. Expected {(num_days_history, len(feature_names_expected))}, got {X_infer_np.shape}")

    X_infer_tensor = torch.from_numpy(X_infer_np).unsqueeze(0) # Add batch dimension: [1, T, D]

    print(f"Inference input tensor shape: {X_infer_tensor.shape}")
    if torch.isnan(X_infer_tensor).any():
        raise ValueError("NaNs detected in the final inference input tensor after scaling!")

    print("--- Inference Data Compilation Complete ---")
    # Return df_full_cleaned.reset_index() for plotting context
    return X_infer_tensor, scaler, start_date, df_full_cleaned.reset_index()


# --- Function for multi-step forecasting ---
def infer_future(model: lstmAlgorithm, start_date_str: str, days_to_predict: int,
                 T_hist: int, scaler_path: str, full_df_path: str,
                 feature_names: list, target_col: str, device):
    """
    Performs multi-step ahead forecasting using the trained LSTM model.

    Args:
        model: The trained LSTM model (must be loaded with weights).
        start_date_str: The first date for prediction (YYYYMMDD).
        days_to_predict: How many days into the future to predict.
        T_hist: The history window size (timestep T used during training).
        scaler_path: Path to the saved scaler.
        full_df_path: Path to the full dataset CSV.
        feature_names (list): List of feature names in the order model expects.
        target_col (str): Name of the target column ('mean_temp').
        device: CPU or CUDA/MPS device.

    Returns:
        list: List of predicted target values (unscaled).
        list: List of corresponding prediction dates (pd.Timestamp).
        pd.DataFrame: Full cleaned dataframe used for context.
    """
    print(f"\n--- Starting Multi-Step Inference ---")
    print(f"Predicting {days_to_predict} days, starting from {start_date_str}")

    model.eval() # Ensure model is in evaluation mode
    model.to(device)

    # 1. Get the initial input sequence and context data
    last_sequence, scaler, current_pred_date, df_context = compile_inference_data(
        start_date_str, T_hist, full_df_path, scaler_path, feature_names
    )
    last_sequence = last_sequence.to(device) # Move initial sequence to device

    predictions_unscaled = []
    prediction_dates = []

    # Check if target column exists in feature list
    if target_col not in feature_names:
        raise ValueError(f"Target column '{target_col}' is not in the provided feature list.")
    target_col_index = feature_names.index(target_col)
    num_features = len(feature_names)

    # --- Multi-step Prediction Loop ---
    for i in range(days_to_predict):
        print(f"\nPredicting for date: {current_pred_date.strftime('%Y-%m-%d')} (Step {i+1}/{days_to_predict})")
        with torch.no_grad():
            # Get prediction (output is scaled target value)
            pred_scaled_tensor = model(last_sequence) # Shape: [1, 1]
            pred_scaled_value = pred_scaled_tensor.cpu().item() # Get scalar value

        # Store the prediction date
        prediction_dates.append(current_pred_date)

        # --- Inverse transform the prediction ---
        # Create a dummy row with zeros, put the scaled prediction in the target column's position
        dummy_row_scaled = np.zeros((1, num_features))
        dummy_row_scaled[0, target_col_index] = pred_scaled_value

        # Inverse transform the entire dummy row
        pred_unscaled_row = scaler.inverse_transform(dummy_row_scaled)
        # Extract the unscaled prediction from the target column's position
        final_pred_unscaled = pred_unscaled_row[0, target_col_index]
        predictions_unscaled.append(final_pred_unscaled)
        print(f"  Predicted {target_col} (unscaled): {final_pred_unscaled:.2f}")

        # --- Prepare the next input sequence ---
        # This is the core of multi-step forecasting: use the prediction as input for the next step.
        # We need to create the feature vector for the day we just predicted.

        # **Strategy:** Use the predicted scaled target value. For other features,
        # carry over the values from the *last known step* in the current input sequence.
        # This is a simplification; more advanced methods could predict all features or use external forecasts.

        # Get the numpy version of the current input sequence (shape: [1, T, D])
        last_sequence_np = last_sequence.cpu().numpy().squeeze(0) # Shape: [T, D]

        # Get the features from the *last day* of the current sequence
        last_day_features_scaled = last_sequence_np[-1, :].copy() # Shape [D]

        # Create the *new* row of scaled features for the day we just predicted (`current_pred_date`)
        # Update the target feature with the predicted scaled value
        new_row_scaled = last_day_features_scaled
        new_row_scaled[target_col_index] = pred_scaled_value
        new_row_scaled = new_row_scaled.reshape(1, num_features) # Shape [1, D]

        # Construct the next sequence: drop the oldest row (index 0), append the new row
        next_sequence_np = np.vstack((last_sequence_np[1:, :], new_row_scaled)) # Shape [T, D]

        # Convert back to tensor, add batch dim, and move to device
        last_sequence = torch.from_numpy(next_sequence_np.astype(np.float32)).unsqueeze(0).to(device) # Shape [1, T, D]

        # Update the date for the next prediction cycle
        current_pred_date += pd.Timedelta(days=1)

    print("\n--- Multi-Step Inference Complete ---")
    return predictions_unscaled, prediction_dates, df_context


# %%
# --- Run Inference ---
print("\n--- Setting up Inference Run ---")
# 1. Load the trained model weights
inference_model = lstmAlgorithm(inputD=D, hiddenD=hidden_dim, layerD=num_layers, outputD=output_dim)
try:
    # Ensure the path matches where weights were saved
    inference_model.load_state_dict(torch.load(model_weights_save_path, map_location=device))
    print(f"Inference model weights loaded successfully from {model_weights_save_path}")
except FileNotFoundError:
     print(f"ERROR: Model weights file not found at {model_weights_save_path}")
     raise SystemExit("Model weights not found. Cannot run inference.")
except Exception as e:
     print(f"Error loading model weights: {e}")
     raise SystemExit("Failed to load model weights.")

# 2. Define parameters for inference
# Make sure this date has T_steps prior days available in your *full* dataset
predict_start_date_str = "20201209"
num_predict_days = 7 # Predict a week ahead

# 3. Specify paths and feature/target names (must match training)
# Use the path determined when loading initial data, or specify directly
full_dataset_path_for_inference = initial_file_path # Use path from initial load
if full_dataset_path_for_inference is None:
     full_dataset_path_for_inference = input("Enter path to the FULL london_weather.csv file again: ")

scaler_path_for_inference = scaler_path_used # Use scaler path from training setup
features_for_inference = feature_names_used # Use feature names from training setup
target_col_for_inference = target_column   # Use target name from training setup

# 4. Run the inference function
try:
    future_preds, future_dates, df_inf_context = infer_future(
        model=inference_model,
        start_date_str=predict_start_date_str,
        days_to_predict=num_predict_days,
        T_hist=T_steps,                     # Use the same T as during training
        scaler_path=scaler_path_for_inference,
        full_df_path=full_dataset_path_for_inference,
        feature_names=features_for_inference,
        target_col=target_col_for_inference,
        device=device
    )

    # 5. Display and Plot Results
    print("\n--- Future Predictions (Multi-Step Ahead) ---")
    results_df = pd.DataFrame({'date': future_dates, 'prediction': future_preds})
    results_df.set_index('date', inplace=True)
    print(results_df)

    # Optional: Plot predictions vs actuals if available for the prediction period
    # Join predictions with the context dataframe (which contains actuals)
    comparison_df = results_df.join(df_inf_context.set_index('date')[target_col_for_inference], how='left')
    comparison_df.rename(columns={target_col_for_inference: 'actual'}, inplace=True) # Rename for clarity

    plt.figure(figsize=(12, 6))
    plt.plot(comparison_df.index, comparison_df['prediction'], label='Predicted Temperature', marker='o', linestyle='--')
    # Only plot actuals if they exist for the predicted dates
    if not comparison_df['actual'].isnull().all():
         plt.plot(comparison_df.index, comparison_df['actual'], label='Actual Temperature', marker='x', linestyle='-')
    else:
         print("\nNote: Actual temperature data not available for the prediction period in the provided context.")

    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.title(f'Temperature Forecast ({target_col_for_inference}) starting {predict_start_date_str} for {num_predict_days} days')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

except ValueError as e:
    print(f"\n--- ERROR during inference setup or execution ---")
    print(e)
    import traceback
    traceback.print_exc() # Print detailed traceback for debugging
except FileNotFoundError as e:
    print(f"\n--- ERROR: File Not Found during inference ---")
    print(e)
except Exception as e:
     print(f"\n--- An unexpected error occurred during inference ---")
     print(e)
     import traceback
     traceback.print_exc()


# %%
# Final cleanup (optional)
print("\nScript finished.")
try:
    if torch.cuda.is_available():
        torch.cuda.empty_cache() # Clear GPU cache if CUDA was used
        print("Cleared CUDA cache.")
except NameError: # Handle case where torch wasn't imported or CUDA not checked
    pass
except Exception as e:
    print(f"Error clearing cache: {e}")