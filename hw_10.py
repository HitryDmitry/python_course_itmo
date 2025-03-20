import pickle
import uvicorn

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# создаем приложение
app = FastAPI()

# загружаем модель из предыдущего задания
with open("rf_fitted.pkl", 'rb') as file:
  model = pickle.load(file)

# Класс для валидации запросов
class ModelRequestData(BaseModel):
  postcode: int
  lat: float
  lon: float
  total_square: int
  rooms: int
  floor: int
  city: str
  source: str

class Result(BaseModel):
  result: float

# функция для получения предсказания от модели
def rg_predict(X_in, model):
  df_cols = ['city', 'source']
  df_1 = X_in.drop(columns=df_cols, axis=1)
  
  # Список всех возможных городов и источников
  all_cities = ['Балашиха',
    'Видное',
    'Дзержинский',
    'Долгопрудный',
    'Ивантеевка',
    'Королёв',
    'Котельники',
    'Красногорск',
    'Лобня',
    'Лыткарино',
    'Люберцы',
    'Москва',
    'Московский',
    'Мытищи',
    'Одинцово',
    'Подольск',
    'Пушкино',
    'Реутов',
    'Химки',
    'Щербинка',
    'Щёлково']

  all_sources = ["Домклик", "Новострой-М", "ЦИАН", "Яндекс.Недвижимость"]

  # Создаем фиктивные переменные с учетом всех возможных категорий
  df_2 = pd.get_dummies(X_in[df_cols], columns=['city', 'source'])
  df_2 = df_2.reindex(columns=[f'city_{city}' for city in all_cities] + [f'source_{source}' for source in all_sources], fill_value=0)

  X_in = pd.concat([df_1, df_2], axis=1, join='inner')

  preds = model.predict(X_in)
  return preds

# получение предсказания моделью через get-запрос по адресу /predict_get - 2 балла
# Для get-запроса параметры должны быть переданы через url, поэтому мы передаем каждый параметр
# для функции predict_with_get отдельно
@app.get("/predict_get", response_model=Result)
def predict_with_get(
  postcode: int,
  lat: float,
  lon: float,
  total_square: int,
  rooms: int,
  floor: int,
  city: str,
  source: str
):
  
  # создаем словарь
  input_data = {
        "postcode": postcode,
        "lat": lat,
        "lon": lon,
        "total_square": total_square,
        "rooms": rooms,
        "floor": floor,
        "city": city,
        "source": source,
    }
  input_df = pd.DataFrame([input_data])
  prediction = rg_predict(input_df, model)
  return Result(result=prediction[0])

# получение предсказания моделью через post-запрос по адресу /predict_post - 2 балла
# Для post-запроса можно передавать параметры в теле запроса
@app.post("/predict_post", response_model=Result)
def predict_with_post(data: ModelRequestData):
  input_data = data.dict()
  print(input_data)
  input_df = pd.DataFrame([input_data])
  print(input_df)
  prediction = rg_predict(input_df, model)
  return Result(result=prediction[0])

# Реализуйте liveness-пробу (health-check) health - 1 балл
@app.get("/heathcheck")
def health():
  return JSONResponse(content={"message":"Healthy"}, status_code=200)

if __name__ == "__main__":
  uvicorn.run(app, host="127.0.0.1", port=5000)