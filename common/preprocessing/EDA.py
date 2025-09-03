import logging
import pandas as pd

def do_eda(df_train:pd.DataFrame):
    logging.info("##########################")
    logging.info("Start EDA")
    print(f"df_train.shape: {df_train.shape}")
    
    # 전체 결측치 개수
    print(f"df_train의 결측치:{df_train.isnull().sum().sum()}")

    #칼럼별 결측치 개수
    print("\n===== 칼럼별 결측치 개수 =====")
    print(df_train.isnull().sum())

    # 칼럼별 결측치 비율
    print("\n===== 칼럼별 결측치 비율(%) =====")
    print((df_train.isnull().mean() * 100).round(2))