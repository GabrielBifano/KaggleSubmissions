import polars as pl

class DatalLoader:
    
    def __init__(self):
        self.folder_path = '../data'
        self.data = {}
        self.county_mapping = {
            "0":"HARJUMAA",
            "1":"HIIUMAA",
            "2":"IDA-VIRUMAA",
            "3":"J\u00c4RVAMAA",
            "4":"J\u00d5GEVAMAA",
            "5":"L\u00c4\u00c4NE-VIRUMAA",
            "6":"L\u00c4\u00c4NEMAA",
            "7":"P\u00c4RNUMAA",
            "8":"P\u00d5LVAMAA",
            "9":"RAPLAMAA",
            "10":"SAAREMAA",
            "11":"TARTUMAA",
            "12":"UNKNOWN",
            "13":"VALGAMAA",
            "14":"VILJANDIMAA",
            "15":"V\u00d5RUMAA"
        }

    def load_data(self):
        self.train()
        self.client()
        self.electricity_prices()
        self.historical_weather()
        self.weather_station_to_county_mapping()
        self.gas_prices()
        self.forecast_weather()
        return self.data
    
    def train(self):
        self.data['train'] = pl.read_csv(self.folder_path + '/train.csv', try_parse_dates=True)
        return self.data['train']

    def client(self):
        self.data['client'] = pl.read_csv(self.folder_path + '/client.csv', try_parse_dates=True)
        return self.data['client']

    def electricity_prices(self):
        self.data['electricity_prices'] = pl.read_csv(self.folder_path + '/electricity_prices.csv', try_parse_dates=True)
        return self.data['electricity_prices']

    def historical_weather(self):
        self.data['historical_weather'] = pl.read_csv(self.folder_path + '/historical_weather.csv', try_parse_dates=True)
        return self.data['historical_weather']

    def weather_station_to_county_mapping(self):
        self.data['weather_station_to_county_mapping'] = pl.read_csv(self.folder_path + '/weather_station_to_county_mapping.csv', try_parse_dates=True)
        return self.data['weather_station_to_county_mapping']

    def gas_prices(self):
        self.data['gas_prices'] = pl.read_csv(self.folder_path + '/gas_prices.csv', try_parse_dates=True)
        return self.data['gas_prices']
    
    def forecast_weather(self):
        self.data['forecast_weather'] = pl.read_csv(self.folder_path + '/forecast_weather.csv', try_parse_dates=True)
        return self.data['forecast_weather']