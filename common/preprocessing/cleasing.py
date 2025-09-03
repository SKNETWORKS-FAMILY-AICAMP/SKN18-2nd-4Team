import numpy as np
import pandas as pd

def __fillna(df_train:pd.DataFrame, df_test:pd.DataFrame): # fillna 는 결측치를 처리하는 함수
    # df_train.isnull() -> 2차원 데이터(True/False)
    # df_train.isnull().sum() -> 1차원 데이터(컬럼별 결측치 합)
    # -> index(컬럼) / value(결측치 수)
    # df_train.isnull().sum()[df_train.isnull().sum() > 0]
    # -> 결측치가 존재하는 것만 조회 -> 1차원 데이터(index(=컬럼) / value(=결측치 수))
    train_none_cols = df_train.isnull().sum()[df_train.isnull().sum() > 0].index # train_none_cols = 결측치가 존재하는 컬럼들만 담길 것
    test_none_cols = df_test.isnull().sum()[df_test.isnull().sum() > 0].index
    # 전체 결측치 컬럼들
    none_cols = list(set(train_none_cols) | set(test_none_cols))
    for col in none_cols: # 결측치 컬럼 리스트
        try:
            # 통계값 추출
            _value = df_train[col].mean() # 수치형
        except:
            _value = df_test[col].mode()[0] # 범주형
        finally:
            #결측치에 통게값 넣기기
            df_train[col].fillna(_value, inplace=True)
            df_test[col].fillna(_value, inplace=True)

    return df_train, df_test

def __dropcols(df_train:pd.DataFrame, df_test:pd.DataFrame, drop_cols:list):
    # drop_cols 리스트에 있는 컬럼들을 제거
    return df_train.drop(drop_cols, axis=1), df_test.drop(drop_cols, axis=1)

# def tmp_func(data):
#     return np.log1p(data)  -> 이 코드는 재사용 가능, lambda로 표현 가능한데 lambda는 일회용

def __transform_cols(df_train:pd.DataFrame, df_test:pd.DataFrame, transform_cols:list):

    for col in transform_cols: # 변환할 컬럼들
        df_train[col] = df_train[col].map(lambda x : np.log1p(x)) # map 앞에는 무조건 1차원 데이터만 가능하고, apply는 2차원 데이터만 가능
        df_test[col] = df_test[col].map(lambda x : np.log1p(x)) # lambda x : np.log1p(x) = def tmp_func(data): return np.log1p(data)

    return df_train, df_test


def do_cleasing(df_train:pd.DataFrame, df_test:pd.DataFrame
                , drop_cols:list,transform_cols:list):
    # 1. row 중복 데이터 제거
    df_train = df_train.drop_duplicates()

    # 2. 결측치 치환(통계값 <- train 데이터의 통계값, test는 쓰면 안된다)
    df_train, df_test = __fillna(df_train, df_test)

    # 3. 필요없는 컬럼 제거
    df_train, df_test = __dropcols(df_train, df_test, drop_cols=drop_cols)

    # 4. 왜도/첨도 처리
    df_train, df_test = __transform_cols(df_train, df_test, transform_cols=transform_cols)

    # 5. 검증
    assert df_train.shape[1] == df_test.shape[1] # 컬럼 수가 같아야함, 안 그러면 오류

    return df_train, df_test