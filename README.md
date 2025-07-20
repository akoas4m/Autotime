# Time Series Forecasting with AutoTimes and GPT2

This README outlines the steps to process a dataset, configure the necessary files, and run a long-term forecasting task using the AutoTimes model with GPT2 integration.

## Prerequisites
- Python 3.x
- PyTorch and required dependencies installed
- Access to the dataset directory (`./dataset/`)
- GPU (optional, specified as `--gpu 0` in the run command)

## Steps

### Step 1: Process the Raw Dataset
- **Task**: Handle missing values in the raw dataset by performing interpolation and save the processed file in the `dataset` directory.
- **Action**:
  - Load the raw dataset (e.g., `OBS_FD01.csv`).
  - Apply interpolation to fill missing values.
  - Save the cleaned dataset as `OBS_FD01_cleaned.csv` in the `./dataset/` directory.

### Step 2: Modify Data Provider Files
- **Task**: Update the data handling scripts in the `data_provider` directory.
- **Action**:
  - Edit `data_factory.py` to include the custom dataset (`OBS_FD01_cleaned.csv`) and ensure it supports the required data format.
  - Modify `data_loader.py` to handle the dataset loading logic for the forecasting task.

### Step 3: Download GPT2 Model
- **Task**: Obtain the GPT2 model for use in preprocessing.
- **Action**:
  - Download the pretrained GPT2 model from a reliable source (e.g., Hugging Face).
  - Ensure the model is placed in an accessible directory for the project.

### Step 4: Modify Preprocess Script
- **Task**: Update `preprocess.py` to use GPT2 for data preprocessing.
- **Action**:
  - Edit `preprocess.py` to integrate the GPT2 model for feature extraction or data transformation.
  - Ensure the script is configured to process the cleaned dataset (`OBS_FD01_cleaned.csv`).

### Step 5: Generate .pt File
- **Task**: Create a `.pt` file containing the preprocessed data.
- **Action**:
  - Run `preprocess.py` to generate the `.pt` file.
  - Verify that the `.pt` file is saved in the appropriate directory (e.g., `./dataset/` or as specified in the preprocessing script).

### Step 6: Run Long-Term Forecasting
- **Task**: Modify the forecasting script and execute the training via the terminal.
- **Action**:
  - Update `exp_long_term_forecasting.py` to configure the forecasting task with the desired model and parameters.
  - Modify `run.py` to ensure compatibility with the AutoTimes_Gpt2 model and custom dataset.
  - Navigate to the project directory in the terminal and run the following command:
    ```bash
    python run.py --task_name long_term_forecast --is_training 1 --model_id test --model AutoTimes_Gpt2 --data custom --root_path ./dataset/ --data_path OBS_FD01_cleaned.csv --checkpoints ./checkpoints/ --seq_len 672 --label_len 576 --test_pred_len 5 --batch_size 128 --learning_rate 0.001 --mlp_hidden_dim 128 --mlp_hidden_layers 2 --cosine False --use_amp False --gpu 0
