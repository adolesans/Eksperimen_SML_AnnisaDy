import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler

# --- Fungsi Pencari File Otomatis ---
def find_dataset(filename):
    print(f"Mencari file '{filename}'...")
    for root, dirs, files in os.walk('.'):
        if filename in files:
            full_path = os.path.join(root, filename)
            print(f"DITEMUKAN: {full_path}")
            return full_path
    
    print("File dataset tidak ditemukan!")
    sys.exit(1)

# --- Fungsi Outliers ---
def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

# --- Fungsi Utama Preprocessing ---
def preprocess_data(input_path, output_path):
    print(f"Memproses data...")
    df = pd.read_csv(input_path)

    # 1. Handling Missing Values
    cat_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History', 'Loan_Amount_Term']
    for col in cat_cols:
        if col in df.columns: df[col] = df[col].fillna(df[col].mode()[0])
    
    if 'LoanAmount' in df.columns: df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())

    # 2. Outlier Removal
    numeric_targets = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
    existing_numeric = [c for c in numeric_targets if c in df.columns]
    df = remove_outliers_iqr(df, existing_numeric)

    # 3. Encoding & Scaling
    df = pd.get_dummies(df, drop_first=True)
    scaler = StandardScaler()
    processed_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # 4. Simpan Hasil
    # Pastikan folder induknya (preprocessing/) ada
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed_df.to_csv(output_path, index=False)
    print(f"SUKSES! Tersimpan di: {output_path}")

if __name__ == "__main__":
    # Cari input otomatis
    input_csv = find_dataset('loan_dataset.csv')
    
    # OUTPUT YANG BENAR (Langsung di dalam folder preprocessing)
    output_csv = 'Preprocessing/loan_clean.csv'
    
    preprocess_data(input_csv, output_csv)
