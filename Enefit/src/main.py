import holidays
import datetime

from data_storage import DataStorage
from features_generator import FeaturesGenerator
from model import Model
import enefit

# from .competition import make_env
# __all__ = ['make_env']


data_storage = DataStorage()
features_generator = FeaturesGenerator(data_storage=data_storage)

df_train_features = features_generator.generate_features(data_storage.df_data)
df_train_features = df_train_features[df_train_features['target'].notnull()]


estonian_holidays = holidays.country_holidays('EE', years=range(2021, 2026))
estonian_holidays = list(estonian_holidays.keys())

def add_holidays_as_binary_features(df):
    df['country_holiday'] = df.apply(lambda row: (datetime.date(row['year'], row['month'], row['day']) in estonian_holidays) * 1, axis=1)
    
    return df

df_train_features = add_holidays_as_binary_features(df_train_features)

model = Model()
model.fit(df_train_features)

# ------------------------------------------------------------------------------------------------------------------
# LAST PART---------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

env = enefit.make_env()
iter_test = env.iter_test()

for (
    df_test, 
    df_new_target, 
    df_new_client, 
    df_new_historical_weather,
    df_new_forecast_weather, 
    df_new_electricity_prices, 
    df_new_gas_prices, 
    df_sample_prediction
) in iter_test:

    data_storage.update_with_new_data(
        df_new_client=df_new_client,
        df_new_gas_prices=df_new_gas_prices,
        df_new_electricity_prices=df_new_electricity_prices,
        df_new_forecast_weather=df_new_forecast_weather,
        df_new_historical_weather=df_new_historical_weather,
        df_new_target=df_new_target
    )
    df_test = data_storage.preprocess_test(df_test)
    
    df_test_features = features_generator.generate_features(df_test)
    df_test_features = add_holidays_as_binary_features(df_test_features)
    df_sample_prediction["target"] = model.predict(df_test_features)
    
    env.predict(df_sample_prediction)

