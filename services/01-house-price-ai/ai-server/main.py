from enum import Enum

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(
    title="AI Service Blueprint API",
    description="Master FastAPI template for AI services",
    version="1.0.0"
)

model = joblib.load("models/house_price_model.pkl")
location_kmeans = joblib.load("models/location_kmeans.pkl")


class Region(str, Enum):
    """Swagger(/docs)에서 위도/경도 숫자 대신 지역명을 선택해 테스트할 수 있도록
    제공하는 캘리포니아 주요 도시 목록. 값은 각 도시의 대표 위도/경도로 변환된다.
    """
    LOS_ANGELES = "Los Angeles"
    SAN_FRANCISCO = "San Francisco"
    SAN_DIEGO = "San Diego"
    SACRAMENTO = "Sacramento"
    SAN_JOSE = "San Jose"
    FRESNO = "Fresno"
    OAKLAND = "Oakland"
    LONG_BEACH = "Long Beach"
    BAKERSFIELD = "Bakersfield"
    RIVERSIDE = "Riverside"
    SANTA_BARBARA = "Santa Barbara"
    STOCKTON = "Stockton"


# 지역명 -> (Latitude, Longitude) 대표 좌표
# (California Housing 데이터의 Latitude/Longitude 범위: 32.54~41.95 / -124.35~-114.31)
REGION_COORDINATES: dict[Region, tuple[float, float]] = {
    Region.LOS_ANGELES: (34.0522, -118.2437),
    Region.SAN_FRANCISCO: (37.7749, -122.4194),
    Region.SAN_DIEGO: (32.7157, -117.1611),
    Region.SACRAMENTO: (38.5816, -121.4944),
    Region.SAN_JOSE: (37.3382, -121.8863),
    Region.FRESNO: (36.7378, -119.7871),
    Region.OAKLAND: (37.8044, -122.2712),
    Region.LONG_BEACH: (33.7701, -118.1937),
    Region.BAKERSFIELD: (35.3733, -119.0187),
    Region.RIVERSIDE: (33.9806, -117.3755),
    Region.SANTA_BARBARA: (34.4208, -119.6982),
    Region.STOCKTON: (37.9577, -121.2908),
}


class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    region: Region


@app.get("/")
def root():
    return {
        "message": "AI Service Blueprint API is running",
        "status": "success"
    }

@app.get("/health")
def health():
    return {
        "status": "OK"
    }

@app.post("/predict")
def predict(features: HouseFeatures):
    input_df = pd.DataFrame([features.model_dump()])

    # region(지역명)을 모델이 실제로 학습한 Latitude/Longitude 숫자로 변환한다.
    # region 컬럼 자체는 모델이 모르는 값이므로 변환 후 제거한다.
    latitude, longitude = REGION_COORDINATES[features.region]
    input_df["Latitude"] = latitude
    input_df["Longitude"] = longitude
    input_df = input_df.drop(columns=["region"])

    # train_model.py에서 학습한 모델은 원본 8개 Feature 외에
    # RoomsPerHousehold, IncomePerRoom, LocationCluster 파생변수도
    # 함께 사용하므로, 예측 전에 동일한 방식으로 계산해서 추가해준다.
    input_df["RoomsPerHousehold"] = (
        input_df["AveRooms"] / input_df["AveOccup"]
    ).replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    input_df["IncomePerRoom"] = (
        input_df["MedInc"] / input_df["AveRooms"]
    ).replace([float("inf"), float("-inf")], 0.0).fillna(0.0)

    # LocationCluster는 train_model.py가 학습 데이터로 fit해 둔
    # location_kmeans 모델을 그대로 불러와 predict()만 적용한다
    # (새로 fit하지 않음 — 학습 시점과 동일한 군집 기준을 유지).
    input_df["LocationCluster"] = location_kmeans.predict(
        input_df[["Latitude", "Longitude"]]
    )

    predicted_price = model.predict(input_df)[0]

    return {
        "predicted_price": round(float(predicted_price), 3)
    }
