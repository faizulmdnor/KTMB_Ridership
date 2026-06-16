import pandas as pd
from datetime import datetime, timedelta

def data_by_service(df):
    df = df.copy()
    service = df.loc[0, 'service']
    ridership = df['ridership'].tolist()
    df['predict_ridership'] = df['ridership'].copy()

    date_predict = []

    # last 5 data
    last5_data = ridership[-5:]

    # last date
    list_date = df['date'].tolist()
    last_date = list_date[-1]
    last_date = datetime.strptime(last_date, '%Y-%m-%d')

    count = 0

    while count <= 4:
        # date predict
        ldate = last_date
        pdate = datetime.strftime(ldate + timedelta(days=count+1), '%Y-%m-%d')

        # moving average
        avg_ridership = round(sum(last5_data[-5:])/5, 0)
        b = avg_ridership
        last5_data.append(b)

        # insert data in dataframe
        df.loc[len(df)] = {
            'date': pdate,
            'service': service,
            'ridership': None,
            'predict_ridership': avg_ridership
        }

        count += 1

    return df


def get_ktmb_data():
    URL_DATA = 'https://storage.data.gov.my/transportation/ktmb/ridership_ktmb_daily.parquet'
    df = pd.read_parquet(URL_DATA)
    if 'date' in df.columns: df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    return df

if __name__ == "__main__":
    df_ridership = get_ktmb_data()
    service_list = df_ridership['service'].unique().tolist()

    for s in service_list:
        df_service = df_ridership[df_ridership['service'] == s].copy()
        df_service.reset_index(drop=True, inplace=True)

        df_ridership_by_service = data_by_service(df_service)

