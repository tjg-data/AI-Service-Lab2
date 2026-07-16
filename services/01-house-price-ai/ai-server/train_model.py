# ------------------------------------------------------------------
# STEP 1. 사용할 라이브러리 불러오기 (import)
# ------------------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error
)

import joblib


# ------------------------------------------------------------------
# STEP 2. 데이터 가져오기 (Load Dataset)
# ------------------------------------------------------------------
#   MedInc      : 해당 지역(Block Group)의 중간 소득 (단위: 만 달러)
#   HouseAge    : 해당 지역 주택들의 중간 연식 (단위: 년)
#   AveRooms    : 가구당 평균 방 개수
#   AveBedrms   : 가구당 평균 침실 개수
#   Population  : 해당 지역의 인구 수
#   AveOccup    : 가구당 평균 거주 인원
#   Latitude    : 위도
#   Longitude   : 경도
#   MedHouseVal : (정답/타겟) 해당 지역의 중간 주택 가격
#                 단위는 "10만 달러(=$100,000)" 입니다.
#                 예) 값이 4.526 이면 실제로는 약 $452,600 을 의미합니다.
# ------------------------------------------------------------------
df = pd.read_csv("data/housing.csv")


# ------------------------------------------------------------------
# STEP 3. Feature(X)와 Target(y) 분리하기
# ------------------------------------------------------------------
X = df.drop(columns=["MedHouseVal"])   # 정답 컬럼을 제외한 나머지 = 입력값(X)
y = df["MedHouseVal"]                  # 정답 컬럼만 따로 = 정답(y)


# ------------------------------------------------------------------
# STEP 4. Train(학습용) / Test(시험용) 데이터 분리하기
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ------------------------------------------------------------------
# STEP 5. Feature Engineering - 파생변수 추가
# ------------------------------------------------------------------
def add_ratio_features(base_df: pd.DataFrame) -> pd.DataFrame:
    """RoomsPerHousehold, IncomePerRoom 파생변수를 추가한 복사본을 반환한다.

    원본 DataFrame(base_df)은 수정하지 않는다.
    """
    engineered_df = base_df.copy()

    engineered_df["RoomsPerHousehold"] = (
        engineered_df["AveRooms"] / engineered_df["AveOccup"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    engineered_df["IncomePerRoom"] = (
        engineered_df["MedInc"] / engineered_df["AveRooms"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return engineered_df


X_train = add_ratio_features(X_train)
X_test = add_ratio_features(X_test)

# LocationCluster: 반드시 학습 데이터(X_train)에만 fit!
location_kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
location_kmeans.fit(X_train[["Latitude", "Longitude"]])

X_train["LocationCluster"] = location_kmeans.predict(X_train[["Latitude", "Longitude"]])
X_test["LocationCluster"] = location_kmeans.predict(X_test[["Latitude", "Longitude"]])  # 검증 데이터는 predict()만!

print("=" * 70)
print("[STEP 6] Feature Engineering 적용 완료")
print("=" * 70)
print(f"학습 데이터 컬럼({len(X_train.columns)}개): {list(X_train.columns)}")
print(X_train[["RoomsPerHousehold", "IncomePerRoom", "LocationCluster"]].head())


# ------------------------------------------------------------------
# STEP 6. 모델(알고리즘) 생성하기 - RandomForestRegressor
# ------------------------------------------------------------------
model = RandomForestRegressor(n_estimators=200, random_state=42)


# ------------------------------------------------------------------
# STEP 7. 학습(Training) - model.fit()
# ------------------------------------------------------------------
model.fit(X_train, y_train)


# ------------------------------------------------------------------
# STEP 8. 예측(Prediction) - model.predict()
# ------------------------------------------------------------------
pred = model.predict(X_test)


# ------------------------------------------------------------------
# STEP 9. 학습된 모델 저장하기 - joblib.dump()
# ------------------------------------------------------------------
joblib.dump(
    model,
    "models/house_price_model.pkl"
)

joblib.dump(
    location_kmeans,
    "models/location_kmeans.pkl"
)


# ------------------------------------------------------------------
# STEP 10. 예측값 vs 실제값 비교해보기
# ------------------------------------------------------------------
result_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predict": pred
})

print("\n" + "=" * 70)
print("[STEP 11] 실제값(Actual) vs 예측값(Predict) 샘플 10개")
print("=" * 70)
print(result_df.head(10))

# Error(오차) = 실제값 - 예측값
#   - 오차가 0에 가까울수록 잘 맞춘 것입니다.
#   - 오차가 양수(+)면 "모델이 실제보다 낮게 예측(과소예측)"했다는
#     뜻이고, 음수(-)면 "모델이 실제보다 높게 예측(과대예측)"했다는
#     뜻입니다.
result_df["Error"] = result_df["Actual"] - result_df["Predict"]

print("\n오차(Error = Actual - Predict) 컬럼을 추가한 결과")
print(result_df.head(10))


# ------------------------------------------------------------------
# STEP 11. 모델 성능 평가 (Evaluation Metrics)
# ------------------------------------------------------------------
r2 = r2_score(y_test, pred)                    # 얼마나 잘 맞았는가? (0~1, 높을수록 좋음)
mae = mean_absolute_error(y_test, pred)        # 평균적으로 얼마나 틀렸는가? (낮을수록 좋음)
rmse = root_mean_squared_error(y_test, pred)   # 큰 오차까지 고려한 평균 오차 (낮을수록 좋음)

print("\n" + "=" * 70)
print("[STEP 12] 모델 성능 평가 결과 (Feature Engineering 적용)")
print("=" * 70)

print(f"R²   : {r2:.4f}")   # Feature Engineering Notebook 기준 약 0.81
print(f"MAE  : {mae:.4f}")  # Feature Engineering Notebook 기준 약 0.33
print(f"RMSE : {rmse:.4f}") # Feature Engineering Notebook 기준 약 0.50

# 위 숫자를 "그냥 숫자"가 아니라 사람이 이해할 수 있는 말로
# 바로 풀어서 보여줍니다. (참고: MedHouseVal의 단위는 10만 달러)
print("\n[쉽게 풀어보면]")
print(f"  - 이 모델은 주택 가격이 왜 다른지를 약 {r2 * 100:.1f}% 정도 설명해내고 있습니다.")
print(f"  - 평균적으로 실제 가격과 약 ${mae * 100_000:,.0f} 정도 차이가 납니다. (MAE 기준)")
print(f"  - 크게 틀린 경우까지 고려하면 평균 오차는 약 ${rmse * 100_000:,.0f} 수준입니다. (RMSE 기준)")
print("  - 더 자세한 지표 설명은 ALGORITHM_GUIDE.md 문서를 참고하세요.")
print("  - Feature Engineering의 전체 검증 과정은")
print("    notebooks/california_housing_feature_engineering.ipynb 문서를 참고하세요.")

# ------------------------------------------------------------------
# STEP 12. FastAPI Swagger 실습을 위한 샘플 데이터 확인
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("[STEP 13] FastAPI Swagger 테스트용 샘플 데이터 (원본 df 앞부분)")
print("=" * 70)
print(df.head())
