import pandas as pd  # 데이터 분석을 위한 핵심 라이브러리. csv/엑셀 등을 DataFrame(표) 형태로 다룰 때 사용.
from sklearn.model_selection import train_test_split  # 데이터 분할 함수. 학습용/평가용 서브셋 생성.
import logging # 로그 출력용 표준 라이브러리. print보다 체계적으로 단계/레벨별 메시지를 관리할 수 있음.

def do_split():
    logging.info("##########################") # 정보(info) 레벨 로그 : 실행 구간을 눈에 띄게 구분하기 위한 구분선 
    logging.info("Start Split Data")  # 정보(INFO) 레벨 로그: "데이터 분할 시작"을 알림. (운영 시 로그로 흐름 추적 용이)
    df = pd.read_csv("./data/E Commerce Dataset.csv")  # - df는 전체 원본 데이터(특징+타깃 포함). 여기서는 'Churn' 컬럼이 타깃(레이블).

# train_test_split으로 전체 DataFrame(df)을 그대로 분할.
    # 핵심 포인트:
    # - X/y로 나누지 않고 df 전체를 넣되,
    # - stratify에 타깃 컬럼(df["Churn"])을 지정하면, train/test 모두에서 'Churn' 비율이 동일하게(층화) 유지됨.
    train, test = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df["Churn"]
    )

    train.to_csv("./data/train.csv", index=False)  # 학습 세트를 CSV로 저장. index=False: 행 인덱스 번호는 파일에 쓰지 않음(불필요한 컬럼 방지).
    test.to_csv("./data/test.csv", index=False)  # 테스트 세트를 CSV로 저장. 실무에서는 보통 test에 타깃 컬럼이 없거나 숨긴 상태로 사용.

    print("train:", train.shape, " test:", test.shape)  # 각 세트의 크기 출력. (행, 열) 튜플 형태. 분할 비율이 의도대로 되었는지 빠르게 확인 가능.
    print("train Churn ratio:\n", train["Churn"].value_counts(normalize=True))  # 학습 세트에서 'Churn' 값의 비율 출력. / value_counts(normalize=True)는 빈도를 비율(합=1.0)로 반환.
    print("test Churn ratio:\n", test["Churn"].value_counts(normalize=True))  # 테스트 세트에서 'Churn' 값의 비율 출력. / 위의 train과 거의 동일해야 stratify가 잘 적용된 것.