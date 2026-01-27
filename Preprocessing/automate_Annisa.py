import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Remove outlier (IQR)
def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

# Preprocessing
def preprocess_data(input_path, output_path):
    # 1. Load Data
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File tidak ditemukan di: {input_path}")
    
    df = pd.read_csv(input_path)
    print(f"Data awal dimuat: {df.shape}")

    # 2. Handling Missing Values
    cat_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History', 'Loan_Amount_Term']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    # Handling numeric missing values
    if 'LoanAmount' in df.columns:
        df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())

    # 3. Encoding (One-Hot Encoding)
    df = pd.get_dummies(df, drop_first=True)

    # 4. Outlier Removal
    numeric_targets = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
    existing_numeric = [c for c in numeric_targets if c in df.columns]
    df_clean = remove_outliers_iqr(df, existing_numeric)

    # 5. Scaling
    target_col = 'Loan_Status_Y' 
    targets = [col for col in df_clean.columns if 'Loan_Status' in col]
    
    if targets:
        target_col = targets[0]
        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        processed_df = pd.DataFrame(X_scaled, columns=X.columns)
        processed_df[target_col] = y.values
    else:
        # Fallback jika target tidak ada/berbeda
        scaler = StandardScaler()
        processed_df = pd.DataFrame(scaler.fit_transform(df_clean), columns=df_clean.columns)

    # 6. Simpan Hasil
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed_df.to_csv(output_path, index=False)
    print(f"Data tersimpan di: {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    
    # Path Input
    input_file = os.path.join(project_root, 'loan_dataset_raw', 'loan_dataset.csv')
    
    # Path Output
    output_file = os.path.join(base_dir, 'loan_dataset_preprocessing', 'loan_clean.csv')
    
    preprocess_data(input_file, output_file)