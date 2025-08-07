#2025-08-04 ~ 2025-08-06 데이터 병합 및 전처리, 모델 학습 + 시각화 코드
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor, VotingRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

#--------------------------------------------------------------------------------------------------------
# 한글 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"
font_prop = fm.FontProperties(fname=font_path)
plt.rc('font', family=font_prop.get_name())
plt.rcParams['axes.unicode_minus'] = False

# 시군명 추출 함수
def extract_city_name(filename: str) -> str:
    name = os.path.basename(filename).replace('.csv', '')
    parts = name.split('_')
    if parts[-1][-1] == '시':      # 형식: ..._YYYYMM_화성시
        return parts[-1]
    elif parts[-2][-1] == '시':    # 형식: ..._수원시_YYYYMM
        return parts[-2]
    else:
        raise ValueError(f"시군명을 찾을 수 없습니다: {filename}")

# CSV 읽기 함수 (UTF-8 → cp949 fallback)
def read_csv_flexible(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding='cp949')

# 데이터 폴더 지정
data_folder = r'your_card_consumption_data_folder'  # 실제 데이터 폴더 경로로 변경
files = glob.glob(os.path.join(data_folder, "*.csv"))

# 전체 데이터 수집
all_data = []

for file in files:
    if not os.path.basename(file).startswith("~$"):
        try:
            city = extract_city_name(file)
            df = read_csv_flexible(file)

            # 날짜 처리
            if 'ta_ymd' not in df.columns:
                print(f"[누락] 'ta_ymd' 없음: {file}")
                continue

            df['ta_ymd'] = pd.to_datetime(df['ta_ymd'], format='%Y%m%d', errors='coerce')
            df['region'] = city

            # 커피/음료 필터링
            df = df[df['card_tpbuz_nm_2'] == '커피/음료']

            all_data.append(df)
            print(f"[로드 완료] {file} - {len(df)} rows")
        except Exception as e:
            print(f"[오류] {file}: {e}")

# 하나로 병합
card_df = pd.concat(all_data, ignore_index=True)

card_df['ym'] = card_df['ta_ymd'].dt.to_period('M').astype(str)


# 지역 + 연월 단위 집계 (총 금액 및 건수)
card_agg = card_df.groupby(['region', 'ym']).agg({
    'amt': 'sum',
    'cnt': 'sum'
}).reset_index().rename(columns={'amt': 'total_amt', 'cnt': 'total_cnt'})

card_agg['avg_amt_per_cnt'] = card_agg['total_amt'] / card_agg['total_cnt']

# ----------------------------------------------------------------------------------

# 2. 영업/폐업 데이터 로드 및 전처리
store_df = pd.read_csv(r'your_cafe_data_file', encoding='cp949')
store_df['인허가일자'] = pd.to_datetime(store_df['인허가일자'], errors='coerce')
store_df['폐업일자'] = pd.to_datetime(store_df['폐업일자'], errors='coerce')
store_df['region'] = store_df['소재지전체주소'].str.extract(r'경기도\s(\S+?)[\s,]')

# 인허가/폐업 월 단위 추출
store_df['open_ym'] = store_df['인허가일자'].dt.to_period('M').astype(str)
store_df['close_ym'] = store_df['폐업일자'].dt.to_period('M').astype(str)

# 지역 + 연월 단위로 개수 집계
open_count = store_df.groupby(['region', 'open_ym']).size().reset_index(name='n_open')
close_count = store_df.groupby(['region', 'close_ym']).size().reset_index(name='n_close')

# ----------------------------------------------------------------------------------

# 3. 브랜드 평판 지수 전처리
brand_df = pd.read_csv(r'your_brand_reputation_file', encoding='cp949')

# 열 이름에서 날짜 추출
value_cols = [col for col in brand_df.columns if '브랜드평판지수' in col]
brand_long = pd.melt(brand_df, id_vars=['순위'], value_vars=value_cols,
                     var_name='ym', value_name='brand_index')

# '2020년8월_브랜드평판지수' → '2020-08'
brand_long['brand'] = pd.melt(brand_df, id_vars=['순위'], value_vars=[col.replace('브랜드평판지수', '커피') for col in value_cols])['value']
brand_long['ym'] = brand_long['ym'].str.extract(r'(\d{4})년(\d{1,2})월').apply(lambda x: f"{x[0]}-{int(x[1]):02}", axis=1)

# 월별 Top1 브랜드 점수만 사용 (또는 평균점수 등도 가능)
brand_top1 = brand_long[brand_long['순위'] == 1][['ym', 'brand_index']].rename(columns={'brand_index': 'top1_brand_index'})

# ----------------------------------------------------------------------------------

# 4. 통합
# 우선 지역 + ym을 기준으로 카드 + 개업 + 폐업
df_merge = card_agg.copy()
df_merge = df_merge.merge(open_count.rename(columns={'open_ym': 'ym'}), on=['region', 'ym'], how='left')
df_merge = df_merge.merge(close_count.rename(columns={'close_ym': 'ym'}), on=['region', 'ym'], how='left')

# 결측값 0으로 처리
df_merge[['n_open', 'n_close']] = df_merge[['n_open', 'n_close']].fillna(0)

# 브랜드 평판 지수는 전체 지역에 공통 적용
df_merge = df_merge.merge(brand_top1, on='ym', how='left')

save_path = r'C:\develop\AI_camp\project2\merged_data.csv'
df_merge.to_csv(save_path, index=False, encoding='utf-8-sig')
print(f"[저장 완료] {save_path}")


#----------------------------------------------------------------



# 데이터 준비
df = df_merge.copy()

# 날짜 변환
df['ym'] = pd.to_datetime(df['ym'], errors='coerce')
df['ym_num'] = df['ym'].dt.year * 100 + df['ym'].dt.month
df['month'] = df['ym'].dt.month

# 계절 파생 변수 (봄:1, 여름:2, 가을:3, 겨울:4)
def get_season(month):
    if month in [3, 4, 5]:
        return 1  # 봄
    elif month in [6, 7, 8]:
        return 2  # 여름
    elif month in [9, 10, 11]:
        return 3  # 가을
    else:
        return 4  # 겨울

df['season'] = df['month'].apply(get_season)

# 폐업률 추가
df['폐업률'] = df['n_close'] / (df['n_open'] + df['n_close'])

# 개폐업 비율
df['open_close_ratio'] = df['n_close'] / (df['n_open'] + 1)

# 지역 기준 정렬 (증감률 파생을 위해)
df.sort_values(['region', 'ym'], inplace=True)

# 증감률 파생 변수
df['open_rate_change'] = df.groupby('region')['n_open'].pct_change()
df['close_rate_change'] = df.groupby('region')['n_close'].pct_change()


df['brand_index_change'] = df['top1_brand_index'].pct_change()


# ----------------------------------------------------------------------------------


# 1. 원두 데이터 불러오기
bean_df = pd.read_csv(r'C:\develop\AI_camp\project2\원두가격비교데이터.csv')

# 컬럼명 정리 (필요시)
bean_df.rename(columns={
    '연도': 'year',
    '아라비카_한잔당(원)': 'arabica_price',
    '로부스타_한잔당(원)': 'robusta_price'
}, inplace=True)

# 7:3 가중 평균 원두 가격 계산
bean_df['weighted_price'] = bean_df['arabica_price'] * 0.7 + bean_df['robusta_price'] * 0.3

# 전년 대비 변화율 (추가 피처용)
bean_df['weighted_price_change'] = bean_df['weighted_price'].pct_change()

# 메인 df에서 연도 추출
df['year'] = df['ym'].dt.year

# 병합
df = df.merge(bean_df[['year', 'weighted_price', 'weighted_price_change']], on='year', how='left')

#-----------------------------------------------------------------------------------

# 최저임금 데이터 불러오기
wage_df = pd.read_csv(r'C:\develop\AI_camp\project2\최저임금_2016_2025.csv')  # 파일 경로 맞게 수정

# 컬럼명 정리
wage_df.rename(columns={
    '연도': 'year',
    '평균최저임금': 'min_wage',
    '전년대비인상률(%)': 'wage_increase'
}, inplace=True)

# 병합
df = df.merge(wage_df, on='year', how='left')

#--------------------------------------------------------------------------------
# 소규모 임대료 불러오기 및 병합 처리
# 소규모 임대료 데이터 불러오기 및 계절 기준 확장

# 1. 불러오기
rent_small_df = pd.read_csv(r'your_small_rent_file', encoding='cp949')
rent_big_df = pd.read_csv(r'your_big_rent_file', encoding='cp949')

# 2. '지역(추출)' 열 기반으로 region 정리 + '시' 붙이기
rent_small_df['region'] = rent_small_df['지역(추출)'].astype(str).str.strip() + '시'
rent_big_df['region'] = rent_big_df['지역(추출)'].astype(str).str.strip() + '시'

# 공백 제거 (열 이름 앞뒤 공백 제거)
rent_big_df.columns = rent_big_df.columns.str.strip()
rent_small_df.columns = rent_small_df.columns.str.strip()

# rent_small 처리
value_cols_small = [col for col in rent_small_df.columns if '년' in col]
rent_small_long = rent_small_df.melt(
    id_vars=['region'],
    value_vars=value_cols_small,
    var_name='year',
    value_name='rent_small'
)

# rent_big 처리
value_cols_big = [col for col in rent_big_df.columns if '년' in col]
rent_big_long = rent_big_df.melt(
    id_vars=['region'],
    value_vars=value_cols_big,
    var_name='year',
    value_name='rent_big'
)


# 4. 연도 추출 및 정수형 변환
rent_small_long['year'] = rent_small_long['year'].str.extract(r'(\d{4})')
rent_small_long = rent_small_long.dropna(subset=['year'])
rent_small_long['year'] = rent_small_long['year'].astype(int)

rent_big_long['year'] = rent_big_long['year'].str.extract(r'(\d{4})')
rent_big_long = rent_big_long.dropna(subset=['year'])
rent_big_long['year'] = rent_big_long['year'].astype(int)


# 5. 계절 확장
season_list = [1, 2, 3, 4]
expanded_rows = []
for _, row in rent_small_long.iterrows():
    for season in season_list:
        expanded_rows.append({
            'region': row['region'],
            'year': row['year'],
            'season': season,
            'rent_small': row['rent_small']
        })
rent_small_season = pd.DataFrame(expanded_rows)

# 동일한 방식으로 대형 임대료도 계절 확장
expanded_rows_big = []
for _, row in rent_big_long.iterrows():
    for season in season_list:
        expanded_rows_big.append({
            'region': row['region'],
            'year': row['year'],
            'season': season,
            'rent_big': row['rent_big']
        })
rent_big_season = pd.DataFrame(expanded_rows_big)

# 6. 메인 df 병합
df = df.merge(rent_small_season, on=['region', 'year', 'season'], how='left')
df = df.merge(rent_big_season, on=['region', 'year', 'season'], how='left')

# ------------------------------------------------------------------------------------
# NaN 평균으로 대체 (상관관계 분석 및 고정비 계산을 위해)
mean_rent_small = df['rent_small'].mean()
df['rent_small'] = df['rent_small'].fillna(mean_rent_small)

mean_rent_big = df['rent_big'].mean()
df['rent_big'] = df['rent_big'].fillna(mean_rent_big)

# 임대료 계산 (면적 가정 적용)
avg_area = 30  # 필요시 수정

df['rent_cost_small'] = df['rent_small'] * avg_area
df['rent_cost_big'] = df['rent_big'] * avg_area




# 원하는 방식에 따라 선택 or 평균
df['avg_rent_cost'] = df[['rent_cost_small', 'rent_cost_big']].mean(axis=1)

# 고정비 계산
df['fixed_cost'] = df['min_wage'] + df['weighted_price'] + df['avg_rent_cost']
#----------------------------------------------------------------------------------------


# 상관분석 대상 수치형 변수 선택
corr_cols = [
    '폐업률',
    'top1_brand_index', 'season', 'weighted_price', 'min_wage',
    'fixed_cost', 'avg_amt_per_cnt', 'avg_rent_cost', 'brand_index_change', 'weighted_price_change', 'wage_increase'
]

df_corr = df[corr_cols].copy()
df_corr = df_corr.fillna(df_corr.mean())

# 상관계수 계산
target_corr = df_corr.corr()['폐업률'].drop('폐업률').sort_values()

# 한국어 변수명 매핑
label_map = {
    'total_amt': '카드 소비 금액',
    'total_cnt': '카드 소비 건수',
    'n_open': '신규 개업 수',
    'n_close': '폐업 수',
    'open_close_ratio': '개폐업 비율',
    'open_rate_change': '개업 증감률',
    'close_rate_change': '폐업 증감률',
    'top1_brand_index': '브랜드 평판지수',
    'ym_num': '연월',
    'season': '계절',
    'brand_index_change': '브랜드 지수 변화율',
    'weighted_price': '평균 원두 가격',
    'weighted_price_change': '원두 가격 변화율',
    'min_wage': '최저임금',
    'wage_increase': '최저임금 인상률',
    'fixed_cost': '고정비',
    'rent_cost_small': '소규모 임대료',
    'avg_amt_per_cnt': '건당 평균 소비 금액',
    'rent_cost_big': '대규모 임대료',
    'avg_rent_cost': '평균 임대료'
}
target_corr.index = target_corr.index.map(label_map)

# 시각화
plt.figure(figsize=(8, 5))
ax = sns.barplot(x=target_corr.values, y=target_corr.index, palette='coolwarm')

# 수치 표시
for i, (value, label) in enumerate(zip(target_corr.values, target_corr.index)):
    plt.text(value + 0.01 * (1 if value >= 0 else -1), i, f'{value:.2f}',
             va='center', ha='left' if value >= 0 else 'right', fontsize=7)

plt.title("폐업률과 변수별 상관계수", fontsize=14)
plt.xlabel("상관계수 (Correlation with 폐업률)")
plt.ylabel("")
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#-------------------------------------------------------------------------------------------
# 8. 히트맵 시각화
corr_matrix = df_corr.corr()
corr_matrix.rename(columns=label_map, index=label_map, inplace=True)

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title("변수 간 상관관계 히트맵", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


#---------------------------------------------------------------------------------------------


# 3. 분석용 피처/타겟 분리
target_col = '폐업률'
non_feature_cols = [
    'total_amt', 'total_cnt', '폐업률', 'open_rate_change', 'close_rate_change', 'brand_index_change', 'rent_big', 'rent_small',
    'weighted_price_change', 'wage_increase', 'rent_cost_small', 'rent_cost_big', 'year', 'ym', 'region', 'ym_num', 'month', 'n_close', 'n_open'
]



# 유일값이 전부 다른 식별자 제거
for col in df.columns:
    if df[col].dtype == object and df[col].nunique() == df.shape[0]:
        non_feature_cols.append(col)

X = df.drop(non_feature_cols, axis=1)
y = df[target_col]

# 4. 범주형 변수 수치 인코딩
for c in X.select_dtypes(include='object').columns:
    X[c] = X[c].astype('category').cat.codes

# 5. 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
#---------------------------------------------------------------------------------------------

def print_metrics(y_true, y_pred, model_name="모델명"):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    print(f"\n {model_name} 성능 평가")
    print(f"  - MAE : {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"  - RMSE: {rmse:.4f}")
    print(f"  - R²  : {r2_score(y_true, y_pred):.4f}")



lgbm = LGBMRegressor(
    n_estimators=500,          # 총 트리 개수 (많을수록 안정적이지만 느림)
    learning_rate=0.03,        # 학습률 (낮을수록 천천히 학습 → 과적합 방지)
    max_depth=4,               # 개별 트리의 최대 깊이 (작게 제한 → 복잡도 감소)
    min_child_samples=30,      # 리프 노드가 가져야 할 최소 샘플 수 (작을수록 과적합 위험 증가)
    subsample=0.7,             # 각 트리 학습 시 사용할 샘플 비율 (샘플링 → 일반화 향상)
    colsample_bytree=0.7,      # 각 트리 학습 시 사용할 피처 비율 (피처 샘플링 → 과적합 방지)
    reg_alpha=0.5,             # L1 정규화 (피처 선택 성격, 과적합 방지)
    reg_lambda=0.5,            # L2 정규화 (가중치 크기를 제한하여 일반화 향상)
    random_state=42,           # 결과 재현성을 위한 시드
    n_jobs=-1                  # 가능한 모든 코어 사용
)


et = ExtraTreesRegressor(
    n_estimators=300,          # 트리 개수
    max_depth=7,               # 트리 최대 깊이 제한
    min_samples_leaf=5,        # 리프 노드의 최소 샘플 수 (너무 작으면 과적합 위험)
    max_features='sqrt',       # 각 노드 분할 시 사용할 피처 개수 (전체 피처의 √개)
    random_state=42            # 시드 고정
)


rf = RandomForestRegressor(
    n_estimators=300,          # 트리 개수
    max_depth=7,               # 트리 깊이 제한
    min_samples_leaf=5,        # 리프 노드 최소 샘플 수
    max_features='sqrt',       # 피처 샘플링 방식 (전체 피처 중 √개)
    random_state=42            # 시드 고정
)
voting = VotingRegressor(estimators=[
    ('lgbm', lgbm),
    ('et', et),
    ('rf', rf)
])

lgbm.fit(X_train, y_train)
y_pred_lgbm = lgbm.predict(X_test)
print_metrics(y_test, y_pred_lgbm, model_name="LightGBM")

# 🔹 ExtraTrees 단독 학습
et.fit(X_train, y_train)
y_pred_et = et.predict(X_test)
print_metrics(y_test, y_pred_et, model_name="ExtraTrees")

# 🔹 RandomForest 단독 학습
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print_metrics(y_test, y_pred_rf, model_name="RandomForest")

voting.fit(X_train, y_train)
y_pred_voting = voting.predict(X_test)

print_metrics(y_test, y_pred_voting, model_name="Voting 앙상블 - L1, L2 규제 추가(과적합 방지)")

# ExtraTreesRegressor 모델 별도 학습 (변수 중요도 추출용)
et.fit(X_train, y_train)

# 변수 중요도 추출 및 정렬
importances = et.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

# 한글 라벨 적용
feature_labels = [label_map.get(f, f) for f in features[indices]]

# 시각화
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=feature_labels, palette='coolwarm')
plt.title("변수 중요도 (ExtraTrees 기준)")
plt.xlabel("중요도")
plt.ylabel("변수명")
plt.tight_layout()
plt.show()


#---------------------------------------------------------------------------------------------
