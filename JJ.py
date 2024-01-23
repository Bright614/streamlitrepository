import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sqlalchemy import create_engine, text, Table, MetaData

connection_string = "mysql+mysqlconnector://admin:Password12!#@database-1.cks69yhf0bnc.ap-northeast-2.rds.amazonaws.com:3306/SSC"
engine = create_engine(connection_string)

connection = engine.connect()
# 메타데이터 생성
metadata = MetaData()

# 테이블 정의
SALES_FORECASTING_DATA_SALES = Table('SALES_FORECASTING_DATA_SALES', metadata, autoload_with=engine)
SALES_FORECASTING_DATA_STORES = Table('SALES_FORECASTING_DATA_STORES', metadata, autoload_with=engine)
SALES_FORECASTING_DATA_TRANSACTIONS = Table('SALES_FORECASTING_DATA_TRANSACTIONS', metadata, autoload_with=engine)

select_query_sales = SALES_FORECASTING_DATA_SALES.select()
select_query_stores = SALES_FORECASTING_DATA_STORES.select()
select_query_transactions = SALES_FORECASTING_DATA_TRANSACTIONS.select()

# SQLAlchemy의 Result 객체를 DataFrame으로 변환
df_sales = pd.read_sql(select_query_sales, connection)
df_stores = pd.read_sql(select_query_stores, connection)
df_transactions = pd.read_sql(select_query_transactions, connection)

#데이터 불러오기
df_holidays = pd.read_excel("C:\\Users\\user\\Desktop\\2023 2학기\\SSC 취업스터디\\존슨앤존슨\\과제\\store_sales_forecasting.xlsx", sheet_name=0)
df_oil = pd.read_excel("C:\\Users\\user\\Desktop\\2023 2학기\\SSC 취업스터디\\존슨앤존슨\\과제\\store_sales_forecasting.xlsx", sheet_name=1)

# 1. 날짜를 datetime 형식으로 변환
df_holidays['date'] = pd.to_datetime(df_holidays['date'])
df_oil['date'] = pd.to_datetime(df_oil['date'])

# 2. 데이터 병합
merged_df = pd.merge(df_sales, df_stores, on='store_nbr')
merged_df = pd.merge(merged_df, df_transactions, on=['date', 'store_nbr'], how='outer')
merged_df['date'] = pd.to_datetime(merged_df['date'])
# 날짜 컬럼을 기준으로 데이터프레임 병합
merged_df = pd.merge(merged_df, df_oil, on='date', how='outer')

# 3. holidays 변수 추가
# 공휴일 유형을 나타내는 집합을 만듭니다
holiday_types = {"Additional", "Bridge", "Event", "Holiday", "Transfer"}

# 날짜가 공휴일인지 여부를 확인하는 조건을 설정합니다
is_holiday_condition = ((df_holidays['type'].isin(holiday_types) & (df_holidays['transferred'] == False)) |
                        ((df_holidays['type'] == "Work Day") & (df_holidays['transferred'] == True)))

# 'date' 열을 기준으로 조건을 merged_df에 병합합니다
merged_df = pd.merge(merged_df, pd.DataFrame({'date': df_holidays['date'], 'is_holiday': is_holiday_condition}),
                     on='date', how='left')
# NaN 값을 0으로 채우고 'is_holiday' 열을 정수형으로 변환합니다
merged_df['is_holiday'] = merged_df['is_holiday'].fillna(0).astype(int)

# 4. dcoilwtico NaN 값을 평균으로 대체
merged_df['dcoilwtico'].fillna(merged_df['dcoilwtico'].mean(), inplace=True)

# 5. 급여일 관련 정보 추가
merged_df['is_payday'] = (merged_df['date'].dt.day == 15) | (merged_df['date'].dt.is_month_end)

# 6. 지진 관련 정보 추가
earthquake_start_date = pd.to_datetime('2016-04-16')
earthquake_effect_duration = pd.DateOffset(weeks=4)
merged_df['is_earthquake'] = (merged_df['date'] >= earthquake_start_date) & (merged_df['date'] <= earthquake_start_date + earthquake_effect_duration)
merged_df['is_earthquake'] = merged_df['is_earthquake'].astype(int)

# 7. family를 더미 변수로 변환
family_dummies = pd.get_dummies(merged_df['family'], prefix='family')
merged_df = pd.concat([merged_df, family_dummies], axis=1)

# 8. 모델 학습 및 예측
features = ['store_nbr', 'transactions', 'dcoilwtico', 'is_holiday', 'onpromotion', 'is_payday', 'is_earthquake'] + list(family_dummies.columns)
X = merged_df[merged_df['isTrain'] == 'Y'][features]
y = merged_df[merged_df['isTrain'] == 'Y']['sales']

# X_train에 str값이 필요하면
X.columns = X.columns.astype(str)

# 데이터를 훈련용과 테스트용으로 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost 모델 생성
model = XGBRegressor(n_estimators=200, max_depth = 7, learning_rate = 0.1, subsample = 0.8,
                     colsample_bytree =0.8, random_state=42)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)
# 음수를 양수로 바꾸기
y_pred_positive = np.abs(y_pred)
# 모델 평가
mse = mean_squared_error(y_test, y_pred_positive)
mae = mean_absolute_error(y_test, y_pred_positive)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_positive)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')

# 예측용 데이터 생성 및 예측
X_pred = merged_df.loc[merged_df['isTrain'] == 'N', features]
X_pred.columns = X_pred.columns.astype(str)
y_pred_new = model.predict(X_pred)
y_prediction = np.abs(y_pred_new)
# 예측값을 DataFrame에 할당
merged_df.loc[merged_df['isTrain'] == 'N', 'sales'] = y_prediction

# 예측값이 있는 항목들을 추출
predicted_sales = merged_df[merged_df['isTrain'] == 'N']
# predicted_sales_no_duplicates 생성
predicted_sales_no_duplicates = predicted_sales.drop_duplicates(subset=['date', 'city', 'state', 'sales'])

# 날짜(date)를 기준으로 도시(city) 및 주(state)에 대한 판매량 합산
grouped_sales_city = predicted_sales_no_duplicates.groupby(['city', 'date', 'state'], as_index=False)['sales'].sum()
grouped_sales_state = predicted_sales_no_duplicates.groupby(['state', 'date'], as_index=False)['sales'].sum()

# 결과 출력
print(grouped_sales_city)
print(grouped_sales_state)

import streamlit as st
import plotly.express as px

# # Streamlit을 이용한 시각화
# st.title('Total Sales Trend')

# # 도시(city)별 판매량 시각화
# st.subheader('Total Sales Trend by City (Predicted)')
# for city in grouped_sales_city['city'].unique():
#     data = grouped_sales_city[grouped_sales_city['city'] == city]
#     st.line_chart(data.set_index('date')['sales'])

# # 주(state)별 판매량 시각화
# st.subheader('Total Sales Trend by State (Predicted)')
# for state in grouped_sales_state['state'].unique():
#     data = grouped_sales_state[grouped_sales_state['state'] == state]
#     st.line_chart(data.set_index('date')['sales'])


header = st.container()


with header:
    st.title("Sales Prediction App With Linear Regression Across Stores")
    st.write("displayed by region : city, state")


st.title('Total Sales Trend')

# 도시(city)별 판매량 시각화
st.subheader('Total Sales Trend by City (Predicted)')

fig_city = px.line(grouped_sales_city, x='date', y='sales', color='city', title='Total Sales Trend by City (Predicted)')
fig_city.update_layout(legend_title_text='City')
st.plotly_chart(fig_city)

# 주(state)별 판매량 시각화
st.subheader('Total Sales Trend by State (Predicted)')

fig_state = px.line(grouped_sales_state, x='date', y='sales', color='state', title='Total Sales Trend by State (Predicted)')
fig_state.update_layout(legend_title_text='State')
st.plotly_chart(fig_state)