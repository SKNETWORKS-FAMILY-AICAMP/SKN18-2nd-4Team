import pandas as pd

def do_encoding(df_train:pd.DataFrame, df_test:pd.DataFrame, encoding_cols:list):
    for col in encoding_cols:
        df_train[col] = df_train[col].astype('category') 
        df_test[col] = df_test[col].astype('category')
        
    return df_train, df_test