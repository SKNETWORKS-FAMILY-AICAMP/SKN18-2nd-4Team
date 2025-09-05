================================================================================
📊 FOOTBALL 데이터 EDA - 통합 분석 결과
================================================================================

## 📁 발견된 CSV 파일 목록:

1.  appearances -> /Users/jina/Documents/GitHub/SKN18-2nd-4Team/data/raw/appearances.csv
2.  club_games -> /Users/jina/Documents/GitHub/SKN18-2nd-4Team/data/raw/club_games.csv
3.  clubs -> /Users/jina/Documents/GitHub/SKN18-2nd-4Team/data/raw/clubs.csv
4.  competitions -> /Users/jina/Documents/GitHub/SKN18-2nd-4Team/data/raw/competitions.csv
5.  game_events -> /Users/jina/Documents/GitHub/SKN18-2nd-4Team/data/raw/game_events.csv
6.  game_lineups -> /Users/jina/Documents/GitHub/SKN18-2nd-4Team/data/raw/game_lineups.csv
7.  games -> /Users/jina/Documents/GitHub/SKN18-2nd-4Team/data/raw/games.csv
8.  player_valuations -> /Users/jina/Documents/GitHub/SKN18-2nd-4Team/data/raw/player_valuations.csv
9.  players -> /Users/jina/Documents/GitHub/SKN18-2nd-4Team/data/raw/players.csv
10. transfers -> /Users/jina/Documents/GitHub/SKN18-2nd-4Team/data/raw/transfers.csv

================================================================================
📋 테이블별 통합 분석
================================================================================

================================================================================
🔍 [APPEARANCES] 테이블 통합 분석
================================================================================

## 📊 2. 테이블 상세 분석

📊 크기: 1,706,806행, 13열 (506.6MB)
📈 데이터 타입: 수치형 9개, 범주형 4개
❌ 결측값: 6개 (0.00%)
📅 날짜 컬럼: ['date']
🔑 ID 컬럼: ['appearance_id', 'game_id', 'player_id', 'player_club_id', 'player_current_club_id', 'competition_id']

📋 컬럼 목록 :

1.  appearance_id (object ) - 유니크: 1,706,806
2.  game_id (int64 ) - 유니크: 66,840
3.  player_id (int64 ) - 유니크: 25,689
4.  player_club_id (int64 ) - 유니크: 1,063
5.  player_current_club_id (int64 ) - 유니크: 438
6.  date (object ) - 유니크: 3,738
7.  player_name (object ) - 유니크: 25,141
8.  competition_id (object ) - 유니크: 43
9.  yellow_cards (int64 ) - 유니크: 3
10. red_cards (int64 ) - 유니크: 2
11. goals (int64 ) - 유니크: 7
12. assists (int64 ) - 유니크: 7
13. minutes_played (int64 ) - 유니크: 122

📄 샘플 데이터 (상위 3행):
appearance_id game_id player_id player_club_id player_current_club_id date player_name competition_id yellow_cards red_cards goals assists minutes_played
0 2231978_38004 2231978 38004 853 235 2012-07-03 Aurélien Joachim CLQ 0 0 2 0 90
1 2233748_79232 2233748 79232 8841 2698 2012-07-05 Ruslan Abyshov ELQ 0 0 0 0 90
2 2234413_42792 2234413 42792 6251 465 2012-07-05 Sander Puri ELQ 0 0 0 0 45

⚠️ 결측값 상세:
player_name : 6개 (0.0%)

## 🏷️ 3. Low 카디널리티 컬럼 분석 (유니크 ≤ 25)

📌 저카디널리티 컬럼 발견:
yellow_cards (int64 ) nunique= 3 -> 0, 1, 2
red_cards (int64 ) nunique= 2 -> 0, 1
goals (int64 ) nunique= 7 -> 0, 1, 2, 3, 4, 5, 6
assists (int64 ) nunique= 7 -> 0, 1, 2, 3, 4, 5, 6

## ⚠️ 4. 데이터 누수 위험 분석

🚨 위험 요소 발견:

1. 날짜/시간 컬럼 존재: ['date'] - 미래 정보 누수 위험
2. 고유값 비율 높음 (100.00%): appearance_id - 식별자 가능성

================================================================================
🔍 [CLUB_GAMES] 테이블 통합 분석
================================================================================

## 📊 2. 테이블 상세 분석

📊 크기: 148,052행, 11열 (35.1MB)
📈 데이터 타입: 수치형 8개, 범주형 3개
❌ 결측값: 93,264개 (5.73%)
🔑 ID 컬럼: ['game_id', 'club_id', 'opponent_id']

📋 컬럼 목록 :

1.  game_id (int64 ) - 유니크: 74,026
2.  club_id (float64 ) - 유니크: 2,829
3.  own_goals (float64 ) - 유니크: 19
4.  own_position (float64 ) - 유니크: 21
5.  own_manager_name (object ) - 유니크: 5,995
6.  opponent_id (float64 ) - 유니크: 2,829
7.  opponent_goals (float64 ) - 유니크: 19
8.  opponent_position (float64 ) - 유니크: 21
9.  opponent_manager_name (object ) - 유니크: 5,995
10. hosting (object ) - 유니크: 2
11. is_win (int64 ) - 유니크: 2

📄 샘플 데이터 (상위 3행):
game_id club_id own_goals own_position own_manager_name opponent_id opponent_goals opponent_position opponent_manager_name hosting is_win
0 2320450 1468.0 0.0 NaN Holger Bachthaler 24.0 2.0 NaN Armin Veh Home 0
1 2320454 222.0 0.0 NaN Volkan Uluc 79.0 2.0 NaN Bruno Labbadia Home 0
2 2320460 1.0 3.0 NaN Jürgen Luginger 86.0 1.0 NaN Robin Dutt Home 1

⚠️ 결측값 상세:
own_position : 44,934개 (30.4%)
opponent_position : 44,934개 (30.4%)
own_manager_name : 1,656개 (1.1%)
opponent_manager_name : 1,656개 (1.1%)
own_goals : 24개 (0.0%)

## 🏷️ 3. Low 카디널리티 컬럼 분석 (유니크 ≤ 25)

📌 저카디널리티 컬럼 발견:
own_goals (float64 ) nunique=19 -> 0.0, 1.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 19.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0
own_position (float64 ) nunique=21 -> 1.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 2.0, 20.0, 21.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0
opponent_goals (float64 ) nunique=19 -> 0.0, 1.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 19.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0
opponent_position (float64 ) nunique=21 -> 1.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 2.0, 20.0, 21.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0
hosting (object ) nunique= 2 -> Away, Home
is_win (int64 ) nunique= 2 -> 0, 1

## ⚠️ 4. 데이터 누수 위험 분석

✅ 위험 요소 없음

================================================================================
🔍 [CLUBS] 테이블 통합 분석
================================================================================

## 📊 2. 테이블 상세 분석

📊 크기: 439행, 17열 (0.3MB)
📈 데이터 타입: 수치형 10개, 범주형 7개
❌ 결측값: 965개 (12.93%)
📅 날짜 컬럼: ['national_team_players', 'stadium_seats']
🔑 ID 컬럼: ['club_id', 'domestic_competition_id']

📋 컬럼 목록 :

1.  club_id (int64 ) - 유니크: 439
2.  club_code (object ) - 유니크: 439
3.  name (object ) - 유니크: 439
4.  domestic_competition_id (object ) - 유니크: 14
5.  total_market_value (float64 ) - 유니크: 0
6.  squad_size (int64 ) - 유니크: 31
7.  average_age (float64 ) - 유니크: 75
8.  foreigners_number (int64 ) - 유니크: 29
9.  foreigners_percentage (float64 ) - 유니크: 167
10. national_team_players (int64 ) - 유니크: 22
11. stadium_name (object ) - 유니크: 420
12. stadium_seats (int64 ) - 유니크: 402
13. net_transfer_record (object ) - 유니크: 305
14. coach_name (float64 ) - 유니크: 0
15. last_season (int64 ) - 유니크: 13
16. filename (object ) - 유니크: 13
17. url (object ) - 유니크: 439

📄 샘플 데이터 (상위 3행):
club_id club_code name domestic_competition_id total_market_value squad_size average_age foreigners_number foreigners_percentage national_team_players stadium_name stadium_seats net_transfer_record coach_name last_season filename url
0 105 sv-darmstadt-98 SV Darmstadt 98 L1 NaN 27 25.6 13 48.1 1 Merck-Stadion am Böllenfalltor 17810 +€3.05m NaN 2023 ../data/raw/transfermarkt-scraper/2023/clubs.json.gz https://www.transfermarkt.co.uk/sv-darmstadt-98/startseite/verein/105
1 11127 ural-ekaterinburg Ural Yekaterinburg RU1 NaN 30 26.5 11 36.7 3 Yekaterinburg Arena 23000 +€880k NaN 2023 ../data/raw/transfermarkt-scraper/2023/clubs.json.gz https://www.transfermarkt.co.uk/ural-ekaterinburg/startseite/verein/11127
2 114 besiktas-istanbul Beşiktaş Jimnastik Kulübü TR1 NaN 30 26.6 15 50.0 8 Beşiktaş Park 42445 €-25.26m NaN 2024 ../data/raw/transfermarkt-scraper/2024/clubs.json.gz https://www.transfermarkt.co.uk/besiktas-istanbul/startseite/verein/114

⚠️ 결측값 상세:
total_market_value : 439개 (100.0%)
coach_name : 439개 (100.0%)
foreigners_percentage : 49개 (11.2%)
average_age : 38개 (8.7%)

## 🏷️ 3. Low 카디널리티 컬럼 분석 (유니크 ≤ 25)

📌 저카디널리티 컬럼 발견:
domestic_competition_id (object ) nunique=14 -> BE1, DK1, ES1, FR1, GB1, GR1, IT1, L1, NL1, PO1, RU1, SC1, TR1, UKR1
total_market_value (float64 ) nunique= 0 ->
national_team_players (int64 ) nunique=22 -> 0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 21, 22, 3, 4, 5, 6, 7, 8, 9
coach_name (float64 ) nunique= 0 ->
last_season (int64 ) nunique=13 -> 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024
filename (object ) nunique=13 -> ../data/raw/transfermarkt-scraper/2012/clubs.json.gz, ../data/raw/transfermarkt-scraper/2013/clubs.json.gz, ../data/raw/transfermarkt-scraper/2014/clubs.json.gz, ../data/raw/transfermarkt-scraper/2015/clubs.json.gz, ../data/raw/transfermarkt-scraper/2016/clubs.json.gz, ../data/raw/transfermarkt-scraper/2017/clubs.json.gz, ../data/raw/transfermarkt-scraper/2018/clubs.json.gz, ../data/raw/transfermarkt-scraper/2019/clubs.json.gz, ../data/raw/transfermarkt-scraper/2020/clubs.json.gz, ../data/raw/transfermarkt-scraper/2021/clubs.json.gz, ../data/raw/transfermarkt-scraper/2022/clubs.json.gz, ../data/raw/transfermarkt-scraper/2023/clubs.json.gz, ../data/raw/transfermarkt-scraper/2024/clubs.json.gz

## ⚠️ 4. 데이터 누수 위험 분석

🚨 위험 요소 발견:

1. 날짜/시간 컬럼 존재: ['national_team_players', 'stadium_seats'] - 미래 정보 누수 위험
2. 고유값 비율 높음 (100.00%): club_code - 식별자 가능성
3. 고유값 비율 높음 (100.00%): name - 식별자 가능성
4. 고유값 비율 높음 (95.67%): stadium_name - 식별자 가능성
5. 고유값 비율 높음 (100.00%): url - 식별자 가능성
6. 높은 결측률 컬럼: ['total_market_value', 'coach_name'] - 데이터 품질 이슈

================================================================================
🔍 [COMPETITIONS] 테이블 통합 분석
================================================================================

## 📊 2. 테이블 상세 분석

📊 크기: 44행, 11열 (0.0MB)
📈 데이터 타입: 수치형 1개, 범주형 9개
❌ 결측값: 16개 (3.31%)
📅 날짜 컬럼: ['confederation', 'is_major_national_league']
🔑 ID 컬럼: ['competition_id', 'country_id']

📋 컬럼 목록 :

1.  competition_id (object ) - 유니크: 44
2.  competition_code (object ) - 유니크: 43
3.  name (object ) - 유니크: 43
4.  sub_type (object ) - 유니크: 12
5.  type (object ) - 유니크: 4
6.  country_id (int64 ) - 유니크: 15
7.  country_name (object ) - 유니크: 14
8.  domestic_league_code (object ) - 유니크: 14
9.  confederation (object ) - 유니크: 1
10. url (object ) - 유니크: 44
11. is_major_national_league (bool ) - 유니크: 2

📄 샘플 데이터 (상위 3행):
competition_id competition_code name sub_type type country_id country_name domestic_league_code confederation url is_major_national_league
0 CIT italy-cup italy-cup domestic_cup domestic_cup 75 Italy IT1 europa https://www.transfermarkt.co.uk/italy-cup/startseite/wettbewerb/CIT False
1 NLSC johan-cruijff-schaal johan-cruijff-schaal domestic_super_cup other 122 Netherlands NL1 europa https://www.transfermarkt.co.uk/johan-cruijff-schaal/startseite/wettbewerb/NLSC False
2 GRP kypello-elladas kypello-elladas domestic_cup domestic_cup 56 Greece GR1 europa https://www.transfermarkt.co.uk/kypello-elladas/startseite/wettbewerb/GRP False

⚠️ 결측값 상세:
country_name : 8개 (18.2%)
domestic_league_code : 8개 (18.2%)

## 🏷️ 3. Low 카디널리티 컬럼 분석 (유니크 ≤ 25)

📌 저카디널리티 컬럼 발견:
sub_type (object ) nunique=12 -> domestic_cup, domestic_super_cup, europa_league, europa_league_qualifying, fifa_club_world_cup, first_tier, league_cup, uefa_champions_league, uefa_champions_league_qualifying, uefa_europa_conference_league, uefa_europa_conference_league_qualifiers, uefa_super_cup
type (object ) nunique= 4 -> domestic_cup, domestic_league, international_cup, other
country_id (int64 ) nunique=15 -> -1, 122, 136, 141, 157, 174, 177, 189, 19, 190, 39, 40, 50, 56, 75
country_name (object ) nunique=14 -> Belgium, Denmark, England, France, Germany, Greece, Italy, Netherlands, Portugal, Russia, Scotland, Spain, Turkey, Ukraine
domestic_league_code (object ) nunique=14 -> BE1, DK1, ES1, FR1, GB1, GR1, IT1, L1, NL1, PO1, RU1, SC1, TR1, UKR1
confederation (object ) nunique= 1 -> europa
is_major_national_league (bool ) nunique= 2 -> False, True

## ⚠️ 4. 데이터 누수 위험 분석

🚨 위험 요소 발견:

1. 날짜/시간 컬럼 존재: ['confederation', 'is_major_national_league'] - 미래 정보 누수 위험
2. 고유값 비율 높음 (100.00%): competition_id - 식별자 가능성
3. 고유값 비율 높음 (97.73%): competition_code - 식별자 가능성
4. 고유값 비율 높음 (97.73%): name - 식별자 가능성
5. 고유값 비율 높음 (100.00%): url - 식별자 가능성

================================================================================
🔍 [GAME_EVENTS] 테이블 통합 분석
================================================================================

## 📊 2. 테이블 상세 분석

📊 크기: 1,035,043행, 10열 (313.1MB)
📈 데이터 타입: 수치형 6개, 범주형 4개
❌ 결측값: 1,502,976개 (14.52%)
📅 날짜 컬럼: ['date']
🔑 ID 컬럼: ['game_event_id', 'game_id', 'club_id', 'player_id', 'player_in_id', 'player_assist_id']

📋 컬럼 목록 :

1.  game_event_id (object ) - 유니크: 1,035,043
2.  date (object ) - 유니크: 3,815
3.  game_id (int64 ) - 유니크: 73,870
4.  minute (int64 ) - 유니크: 121
5.  type (object ) - 유니크: 4
6.  club_id (int64 ) - 유니크: 2,721
7.  player_id (int64 ) - 유니크: 63,004
8.  description (object ) - 유니크: 8,497
9.  player_in_id (float64 ) - 유니크: 56,074
10. player_assist_id (float64 ) - 유니크: 24,596

📄 샘플 데이터 (상위 3행):
game_event_id date game_id minute type club_id player_id description player_in_id player_assist_id
0 2f41da30c471492e7d4a984951671677 2012-08-05 2211607 77 Cards 610 4425 1. Yellow card , Mass confrontation NaN NaN
1 a72f7186d132775f234d3e2f7bc0ed5b 2012-08-05 2211607 77 Cards 383 33210 1. Yellow card , Mass confrontation NaN NaN
2 b2d721eaed4692a5c59a92323689ef18 2012-08-05 2211607 3 Goals 383 36500 , Header, 1. Tournament Goal Assist: , Corner, 1. Tournament Assist NaN 56416.0

⚠️ 결측값 상세:
player_assist_id : 878,284개 (84.9%)
player_in_id : 537,365개 (51.9%)
description : 87,327개 (8.4%)

## 🏷️ 3. Low 카디널리티 컬럼 분석 (유니크 ≤ 25)

📌 저카디널리티 컬럼 발견:
type (object ) nunique= 4 -> Cards, Goals, Shootout, Substitutions

## ⚠️ 4. 데이터 누수 위험 분석

🚨 위험 요소 발견:

1. 날짜/시간 컬럼 존재: ['date'] - 미래 정보 누수 위험
2. 고유값 비율 높음 (100.00%): game_event_id - 식별자 가능성
3. 높은 결측률 컬럼: ['player_in_id', 'player_assist_id'] - 데이터 품질 이슈

================================================================================
🔍 [GAME_LINEUPS] 테이블 통합 분석
================================================================================

## 📊 2. 테이블 상세 분석

📊 크기: 2,191,911행, 10열 (863.8MB)
📈 데이터 타입: 수치형 4개, 범주형 6개
❌ 결측값: 3개 (0.00%)
📅 날짜 컬럼: ['date']
🔑 ID 컬럼: ['game_lineups_id', 'game_id', 'player_id', 'club_id']

📋 컬럼 목록 :

1.  game_lineups_id (object ) - 유니크: 2,191,911
2.  date (object ) - 유니크: 3,059
3.  game_id (int64 ) - 유니크: 57,571
4.  player_id (int64 ) - 유니크: 85,235
5.  club_id (int64 ) - 유니크: 2,464
6.  player_name (object ) - 유니크: 82,154
7.  type (object ) - 유니크: 2
8.  position (object ) - 유니크: 17
9.  number (object ) - 유니크: 203
10. team_captain (int64 ) - 유니크: 2

📄 샘플 데이터 (상위 3행):
game_lineups_id date game_id player_id club_id player_name type position number team_captain
0 b2dbe01c3656b06c8e23e9de714e26bb 2013-07-27 2317258 1443 610 Christian Poulsen substitutes Defensive Midfield 5 0
1 b50a3ec6d52fd1490aab42042ac4f738 2013-07-27 2317258 5017 610 Niklas Moisander starting_lineup Centre-Back 4 0
2 7d890e6d0ff8af84b065839966a0ec81 2013-07-27 2317258 9602 1090 Maarten Martens substitutes Left Winger 11 0

⚠️ 결측값 상세:
position : 3개 (0.0%)

## 🏷️ 3. Low 카디널리티 컬럼 분석 (유니크 ≤ 25)

📌 저카디널리티 컬럼 발견:
type (object ) nunique= 2 -> starting_lineup, substitutes
position (object ) nunique=17 -> Attack, Attacking Midfield, Central Midfield, Centre-Back, Centre-Forward, Defender, Defensive Midfield, Goalkeeper, Left Midfield, Left Winger, Left-Back, Right Midfield, Right Winger, Right-Back, Second Striker, Sweeper, midfield
team_captain (int64 ) nunique= 2 -> 0, 1

## ⚠️ 4. 데이터 누수 위험 분석

🚨 위험 요소 발견:

1. 날짜/시간 컬럼 존재: ['date'] - 미래 정보 누수 위험
2. 고유값 비율 높음 (100.00%): game_lineups_id - 식별자 가능성

================================================================================
🔍 [GAMES] 테이블 통합 분석
================================================================================

## 📊 2. 테이블 상세 분석

📊 크기: 74,026행, 23열 (71.3MB)
📈 데이터 타입: 수치형 9개, 범주형 14개
❌ 결측값: 95,580개 (5.61%)
📅 날짜 컬럼: ['date', 'attendance', 'home_club_formation', 'away_club_formation', 'aggregate']
🔑 ID 컬럼: ['game_id', 'competition_id', 'home_club_id', 'away_club_id']

📋 컬럼 목록 :

1.  game_id (int64 ) - 유니크: 74,026
2.  competition_id (object ) - 유니크: 44
3.  season (int64 ) - 유니크: 13
4.  round (object ) - 유니크: 117
5.  date (object ) - 유니크: 3,826
6.  home_club_id (float64 ) - 유니크: 2,542
7.  away_club_id (float64 ) - 유니크: 2,219
8.  home_club_goals (float64 ) - 유니크: 17
9.  away_club_goals (float64 ) - 유니크: 18
10. home_club_position (float64 ) - 유니크: 21
11. away_club_position (float64 ) - 유니크: 21
12. home_club_manager_name (object ) - 유니크: 5,205
13. away_club_manager_name (object ) - 유니크: 4,870
14. stadium (object ) - 유니크: 2,421
15. attendance (float64 ) - 유니크: 29,712
16. referee (object ) - 유니크: 2,542
17. url (object ) - 유니크: 74,026
18. home_club_formation (object ) - 유니크: 53
19. away_club_formation (object ) - 유니크: 68
20. home_club_name (object ) - 유니크: 439
21. away_club_name (object ) - 유니크: 439
22. aggregate (object ) - 유니크: 117
23. competition_type (object ) - 유니크: 4

📄 샘플 데이터 (상위 3행):
game_id competition_id season round date home_club_id away_club_id home_club_goals away_club_goals home_club_position away_club_position home_club_manager_name away_club_manager_name stadium attendance referee url home_club_formation away_club_formation home_club_name away_club_name aggregate competition_type
0 2321027 L1 2013 1. Matchday 2013-08-11 33.0 41.0 3.0 3.0 8.0 9.0 Jens Keller Thorsten Fink Veltins-Arena 61973.0 Manuel Gräfe https://www.transfermarkt.co.uk/fc-schalke-04_hamburger-sv/index/spielbericht/2321027 4-2-3-1 4-2-3-1 FC Schalke 04 Hamburger SV 3:3 domestic_league
1 2321033 L1 2013 1. Matchday 2013-08-10 23.0 86.0 0.0 1.0 13.0 7.0 Torsten Lieberknecht Robin Dutt EINTRACHT-Stadion 23000.0 Deniz Aytekin https://www.transfermarkt.co.uk/eintracht-braunschweig_sv-werder-bremen/index/spielbericht/2321033 4-3-2-1 4-3-1-2 Eintracht Braunschweig Sportverein Werder Bremen von 1899 0:1 domestic_league
2 2321044 L1 2013 2. Matchday 2013-08-18 16.0 23.0 2.0 1.0 1.0 15.0 Jürgen Klopp Torsten Lieberknecht SIGNAL IDUNA PARK 80200.0 Peter Sippel https://www.transfermarkt.co.uk/borussia-dortmund_eintracht-braunschweig/index/spielbericht/2321044 4-2-3-1 4-3-2-1 Borussia Dortmund Eintracht Braunschweig 2:1 domestic_league

⚠️ 결측값 상세:
home_club_position : 22,467개 (30.4%)
away_club_position : 22,467개 (30.4%)
home_club_name : 12,850개 (17.4%)
away_club_name : 11,455개 (15.5%)
attendance : 9,948개 (13.4%)

## 🏷️ 3. Low 카디널리티 컬럼 분석 (유니크 ≤ 25)

📌 저카디널리티 컬럼 발견:
season (int64 ) nunique=13 -> 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024
home_club_goals (float64 ) nunique=17 -> 0.0, 1.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 17.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0
away_club_goals (float64 ) nunique=18 -> 0.0, 1.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 19.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0
home_club_position (float64 ) nunique=21 -> 1.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 2.0, 20.0, 21.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0
away_club_position (float64 ) nunique=21 -> 1.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 2.0, 20.0, 21.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0
competition_type (object ) nunique= 4 -> domestic_cup, domestic_league, international_cup, other

## ⚠️ 4. 데이터 누수 위험 분석

🚨 위험 요소 발견:

1. 날짜/시간 컬럼 존재: ['date', 'attendance', 'home_club_formation', 'away_club_formation', 'aggregate'] - 미래 정보 누수 위험
2. 고유값 비율 높음 (100.00%): url - 식별자 가능성

================================================================================
🔍 [PLAYER_VALUATIONS] 테이블 통합 분석
================================================================================

## 📊 2. 테이블 상세 분석

📊 크기: 496,606행, 5열 (63.9MB)
📈 데이터 타입: 수치형 3개, 범주형 2개
❌ 결측값: 0개 (0.00%)
📅 날짜 컬럼: ['date']
🔑 ID 컬럼: ['player_id', 'current_club_id', 'player_club_domestic_competition_id']

📋 컬럼 목록 :

1.  player_id (int64 ) - 유니크: 31,078
2.  date (object ) - 유니크: 5,316
3.  market_value_in_eur (int64 ) - 유니크: 425
4.  current_club_id (int64 ) - 유니크: 437
5.  player_club_domestic_competition_id (object ) - 유니크: 14

📄 샘플 데이터 (상위 3행):
player_id date market_value_in_eur current_club_id player_club_domestic_competition_id
0 405973 2000-01-20 150000 3057 BE1
1 342216 2001-07-20 100000 1241 SC1
2 3132 2003-12-09 400000 126 TR1

✅ 결측값 없음

## 🏷️ 3. Low 카디널리티 컬럼 분석 (유니크 ≤ 25)

📌 저카디널리티 컬럼 발견:
player_club_domestic_competition_id (object ) nunique=14 -> BE1, DK1, ES1, FR1, GB1, GR1, IT1, L1, NL1, PO1, RU1, SC1, TR1, UKR1

## ⚠️ 4. 데이터 누수 위험 분석

🚨 위험 요소 발견:

1. 날짜/시간 컬럼 존재: ['date'] - 미래 정보 누수 위험

================================================================================
🔍 [PLAYERS] 테이블 통합 분석
================================================================================

## 📊 2. 테이블 상세 분석

📊 크기: 32,601행, 23열 (36.3MB)
📈 데이터 타입: 수치형 6개, 범주형 17개
❌ 결측값: 43,874개 (5.85%)
📅 날짜 컬럼: ['date_of_birth', 'contract_expiration_date']
🔑 ID 컬럼: ['player_id', 'current_club_id', 'current_club_domestic_competition_id']

📋 컬럼 목록 :

1.  player_id (int64 ) - 유니크: 32,601
2.  first_name (object ) - 유니크: 7,030
3.  last_name (object ) - 유니크: 23,795
4.  name (object ) - 유니크: 31,892
5.  last_season (int64 ) - 유니크: 13
6.  current_club_id (int64 ) - 유니크: 437
7.  player_code (object ) - 유니크: 31,852
8.  country_of_birth (object ) - 유니크: 185
9.  city_of_birth (object ) - 유니크: 8,578
10. country_of_citizenship (object ) - 유니크: 183
11. date_of_birth (object ) - 유니크: 9,306
12. sub_position (object ) - 유니크: 13
13. position (object ) - 유니크: 5
14. foot (object ) - 유니크: 3
15. height_in_cm (float64 ) - 유니크: 53
16. contract_expiration_date (object ) - 유니크: 119
17. agent_name (object ) - 유니크: 2,897
18. image_url (object ) - 유니크: 26,854
19. url (object ) - 유니크: 32,601
20. current_club_domestic_competition_id (object ) - 유니크: 14
21. current_club_name (object ) - 유니크: 437
22. market_value_in_eur (float64 ) - 유니크: 130
23. highest_market_value_in_eur (float64 ) - 유니크: 205

📄 샘플 데이터 (상위 3행):
player_id first_name last_name name last_season current_club_id player_code country_of_birth city_of_birth country_of_citizenship date_of_birth sub_position position foot height_in_cm contract_expiration_date agent_name image_url url current_club_domestic_competition_id current_club_name market_value_in_eur highest_market_value_in_eur
0 10 Miroslav Klose Miroslav Klose 2015 398 miroslav-klose Poland Opole Germany 1978-06-09 00:00:00 Centre-Forward Attack right 184.0 NaN ASBW Sport Marketing https://img.a.transfermarkt.technology/portrait/header/10-1448468291.jpg?lm=1 https://www.transfermarkt.co.uk/miroslav-klose/profil/spieler/10 IT1 Società Sportiva Lazio S.p.A. 1000000.0 30000000.0
1 26 Roman Weidenfeller Roman Weidenfeller 2017 16 roman-weidenfeller Germany Diez Germany 1980-08-06 00:00:00 Goalkeeper Goalkeeper left 190.0 NaN Neubauer 13 GmbH https://img.a.transfermarkt.technology/portrait/header/26-1502448725.jpg?lm=1 https://www.transfermarkt.co.uk/roman-weidenfeller/profil/spieler/26 L1 Borussia Dortmund 750000.0 8000000.0
2 65 Dimitar Berbatov Dimitar Berbatov 2015 1091 dimitar-berbatov Bulgaria Blagoevgrad Bulgaria 1981-01-30 00:00:00 Centre-Forward Attack NaN NaN NaN CSKA-AS-23 Ltd. https://img.a.transfermarkt.technology/portrait/header/65-1683670068.jpg?lm=1 https://www.transfermarkt.co.uk/dimitar-berbatov/profil/spieler/65 GR1 Panthessalonikios Athlitikos Omilos Konstantinoupoliton 1000000.0 34500000.0

⚠️ 결측값 상세:
agent_name : 16,019개 (49.1%)
contract_expiration_date : 12,091개 (37.1%)
country_of_birth : 2,799개 (8.6%)
foot : 2,536개 (7.8%)
city_of_birth : 2,455개 (7.5%)

## 🏷️ 3. Low 카디널리티 컬럼 분석 (유니크 ≤ 25)

📌 저카디널리티 컬럼 발견:
last_season (int64 ) nunique=13 -> 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024
sub_position (object ) nunique=13 -> Attacking Midfield, Central Midfield, Centre-Back, Centre-Forward, Defensive Midfield, Goalkeeper, Left Midfield, Left Winger, Left-Back, Right Midfield, Right Winger, Right-Back, Second Striker
position (object ) nunique= 5 -> Attack, Defender, Goalkeeper, Midfield, Missing
foot (object ) nunique= 3 -> both, left, right
current_club_domestic_competition_id (object ) nunique=14 -> BE1, DK1, ES1, FR1, GB1, GR1, IT1, L1, NL1, PO1, RU1, SC1, TR1, UKR1

## ⚠️ 4. 데이터 누수 위험 분석

🚨 위험 요소 발견:

1. 날짜/시간 컬럼 존재: ['date_of_birth', 'contract_expiration_date'] - 미래 정보 누수 위험
2. 고유값 비율 높음 (97.83%): name - 식별자 가능성
3. 고유값 비율 높음 (97.70%): player_code - 식별자 가능성
4. 고유값 비율 높음 (100.00%): url - 식별자 가능성

================================================================================
🔍 [TRANSFERS] 테이블 통합 분석
================================================================================

## 📊 2. 테이블 상세 분석

📊 크기: 79,646행, 10열 (25.9MB)
📈 데이터 타입: 수치형 5개, 범주형 5개
❌ 결측값: 58,031개 (7.29%)
📅 날짜 컬럼: ['transfer_date']
🔑 ID 컬럼: ['player_id', 'from_club_id', 'to_club_id']

📋 컬럼 목록 :

1.  player_id (int64 ) - 유니크: 10,448
2.  transfer_date (object ) - 유니크: 4,186
3.  transfer_season (object ) - 유니크: 34
4.  from_club_id (int64 ) - 유니크: 9,711
5.  to_club_id (int64 ) - 유니크: 7,367
6.  from_club_name (object ) - 유니크: 10,123
7.  to_club_name (object ) - 유니크: 7,777
8.  transfer_fee (float64 ) - 유니크: 987
9.  market_value_in_eur (float64 ) - 유니크: 176
10. player_name (object ) - 유니크: 10,359

📄 샘플 데이터 (상위 3행):
player_id transfer_date transfer_season from_club_id to_club_id from_club_name to_club_name transfer_fee market_value_in_eur player_name
0 16136 2026-07-01 26/27 417 123 OGC Nice Retired NaN 500000.0 Dante
1 1138758 2026-07-01 26/27 336 631 Sporting CP Chelsea 52140000.0 45000000.0 Geovany Quenda
2 195778 2026-06-30 25/26 79 27 VfB Stuttgart Bayern Munich 0.0 12000000.0 Alexander Nübel

⚠️ 결측값 상세:
market_value_in_eur : 30,316개 (38.1%)
transfer_fee : 27,715개 (34.8%)

## 🏷️ 3. Low 카디널리티 컬럼 분석 (유니크 ≤ 25)

✅ 유니크 ≤ 25인 컬럼이 없습니다.

## ⚠️ 4. 데이터 누수 위험 분석

🚨 위험 요소 발견:

1. 날짜/시간 컬럼 존재: ['transfer_date'] - 미래 정보 누수 위험
