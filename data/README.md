# ⚽ Football Transfer Prediction Data Directory

이 폴더는 Football Transfer Prediction 프로젝트에서 사용하는 모든 데이터 파일들을 포함합니다.

## 📂 폴더 구조

```
data/
├── raw/           # 원본 데이터 파일들 (Transfermarkt 데이터)
├── interim/       # 중간 처리 단계의 데이터 파일들
├── processed/     # 전처리된 데이터 파일들
├── curated/       # 최종 분석용 통합 데이터
└── README.md      # 이 파일
```

## 📋 폴더별 설명

### `raw/`

- **용도**: Transfermarkt 원본 데이터 파일 저장
- **파일**:
  - `appearances.csv` - 선수별 출전 기록
  - `club_games.csv` - 클럽별 경기 결과
  - `clubs.csv` - 클럽 정보
  - `competitions.csv` - 대회 정보
  - `game_events.csv` - 경기 이벤트 데이터
  - `game_lineups.csv` - 경기 라인업
  - `games.csv` - 경기 정보
  - `player_valuations.csv` - 선수 시장가치 변동
  - `players.csv` - 선수 기본 정보
  - `transfers.csv` - 이적 기록
- **특징**:
  - 수정하지 않는 원본 데이터
  - CSV 파일들은 Git에서 제외 (용량 고려)

### `interim/`

- **용도**: 중간 처리 단계의 데이터 파일 저장
- **파일**:
  - 데이터 병합 중간 결과
  - 피쳐 엔지니어링 중간 단계 파일들
  - 결측값 처리 중간 결과
- **특징**:
  - 임시 파일들
  - 필요시 삭제 가능

### `processed/`

- **용도**: 전처리 완료된 데이터 파일 저장
- **파일**:
  - 전처리된 CSV 파일들
  - 피쳐 엔지니어링 완료된 데이터
- **특징**:
  - 분석 및 모델링에 바로 사용 가능
  - 재생성 가능한 파일들

### `curated/`

- **용도**: 최종 분석용 통합 데이터 저장
- **파일**:
  - `player_final.csv` - 최종 통합 데이터셋
- **특징**:
  - 모든 분석의 기준이 되는 마스터 데이터
  - 피쳐 엔지니어링 및 전처리 완료
  - 모델링에 바로 사용 가능

## 🔧 데이터 파일 경로 설정

프로젝트에서 데이터 파일을 사용할 때는 다음과 같이 경로를 설정하세요:

```python
# Python 스크립트에서 (프로젝트 루트에서 실행)
data_path = "data/curated/player_final.csv"

# 주피터 노트북에서 (notes 폴더에서 실행 시)
data_path = "../data/curated/player_final.csv"

# 원본 데이터 사용 시
raw_data_path = "data/raw/players.csv"
```

## 📊 데이터 파일 정보

### player_final.csv (최종 통합 데이터셋)

- **크기**: 약 2.5MB
- **행 수**: 6,910개 선수-시즌 기록
- **열 수**: 24개 피쳐
- **기간**: 2012/13 ~ 2022/23 시즌
- **이적률**: 12.8% (훈련 데이터), 29.6% (테스트 데이터)

#### 주요 피쳐:

- **선수 정보**: `player_name`, `position`, `age_at_season`, `height_in_cm`
- **성과 지표**: `goals`, `assists`, `yellow_cards`, `red_cards`, `season_avg_minutes`
- **시장가치**: `player_market_value_in_eur`, `player_highest_market_value_in_eur`
- **클럽 정보**: `club_name`, `club_squad_size`, `club_average_age`
- **팀 성과**: `season_win_count`, `club_foreigners_percentage`
- **이적 정보**: `transfer` (타겟 변수)

### 원본 데이터 파일들 (raw/)

- **appearances.csv**: 선수별 출전 기록 (경기 수, 출전 시간 등)
- **players.csv**: 선수 기본 정보 (이름, 포지션, 생년월일, 키 등)
- **transfers.csv**: 이적 기록 (이적일, 이적료, 이전/이후 클럽)
- **player_valuations.csv**: 시장가치 변동 기록
- **clubs.csv**: 클럽 정보 (이름, 리그, 국가 등)
- **games.csv**: 경기 정보 (날짜, 홈/어웨이, 결과 등)
- **club_games.csv**: 클럽별 경기 결과 통계

## 🚀 사용 가이드

1. **원본 데이터**: `raw/` 폴더의 파일은 수정하지 마세요
2. **분석용 데이터**: `curated/` 폴더의 `player_final.csv`를 사용하세요
3. **전처리**: `processed/` 폴더에 결과물을 저장하세요
4. **임시 파일**: `interim/` 폴더를 활용하세요
5. **경로 설정**: 상대 경로를 사용하여 호환성을 유지하세요

## ⚠️ 주의사항

- CSV 파일들은 Git에서 제외되어 있습니다 (용량 고려)
- 데이터를 다운로드한 후 `raw/` 폴더에 저장하세요
- `curated/` 폴더의 데이터는 전처리 스크립트로 생성됩니다
