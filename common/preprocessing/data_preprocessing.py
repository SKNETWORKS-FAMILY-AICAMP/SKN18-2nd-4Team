import pandas as pd
from .cleasing import do_cleasing
from .encoding import do_encoding

def do_preprocessing(df_train :pd.DataFrame, df_test: pd.DataFrame, drop_cols:list,
transform_cols:list, encoding_cols:list):
    """
    모델 학습 전 데이터 전처리 함수
    """
    
    # 1. cleasing
    df_train, df_test = do_cleasing(df_train, df_test, drop_cols, transform_cols)



    # 2. change datatype for encoding
    df_train, df_test = do_encoding(df_train, df_test, encoding_cols)

    # 3. 

    return df_train, df_test