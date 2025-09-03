import warnings
warnings.filterwarnings('ignore') # 경고 문구 무시

import logging
logging.basicConfig(level=logging.INFO)

import pandas as pd

from common.dataset import load_dataset
from common.preprocessing.EDA import do_eda
from common.preprocessing.split import do_split
from common.preprocessing.cleasing import __fillna


def modeling():
    logging.info("##########################")
    logging.info("Start Load Dataset")

    # 0. split
    do_split()
    
    #1. train dataset, test dataset 로드
    df_train, df_train_target = load_dataset("./data/train.csv", target_col="Churn")
    df_test, _ = load_dataset("./data/test.csv") # test는 target(정답)이 없으므로 None 처리해주는게 맞음

    # 2. cleasing
    df_train, df_test = __fillna(df_train, df_test)

    # 3. EDA
    do_eda(df_train=df_train)

    # 4. 이상치 처리
    
    # 5. feature engineering

    # 6. modeling

    # 7. training

    # 8. Evaluation

    # 9. Submission

if __name__ == "__main__":
    modeling()