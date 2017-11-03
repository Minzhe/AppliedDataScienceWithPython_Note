import pandas as pd
import numpy as np

doc = []
with open('dates.txt') as file:
    for line in file:
        doc.append(line)

df = pd.Series(doc)

date1 = df.str.extractall(r'(\d{1,2})[/-](\d{1,2})[/-](\d{2}|\d{4})\b')    # 9/27/75
date1.index = date1.index.droplevel(1)
date1_idx = date1.index.values

date2 = df.str.extractall(r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)[-.]? (\d{1,2})[,-]? (\d{4})')    # April 11, 1990
date2.index = date2.index.droplevel(1)
date2_idx = date2.index.values

date3 = df.str.extractall(r'(\d{1,2})? ?((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)[, ]? (\d{4})')    # 24 Jan 2001
date3.index = date3.index.droplevel(1)
date3_idx = date3.index.values

date4 = df.str.extractall(r'(\d{1,2})[/](\d{4})')    # 6/1989
date4.index = date4.index.droplevel(1)
date4_idx = date4.index.values
date4 = date4.loc[list(set(date4_idx) - set(date1_idx))]
date4_idx = date4.index.values

date5_1 = df.str.extractall(r'[^0-9/-](\d{4})[^0-9]')    # 1986
date5_2 = df.str.extractall(r'^(\d{4})[^0-9]')
date5 = pd.concat([date5_1, date5_2])
date5.index = date5.index.droplevel(1)
date5_idx = date5.index.values
date5 = date5.loc[list(set(date5_idx) - set(date2_idx) - set(date3_idx) - set(date4_idx))]


date1.columns = ['month', 'day', 'year']
date1.year = date1.year.apply(lambda x: '19' + x if len(x) <= 2 else x)

date2.columns = ['month', 'day', 'year']

date3.columns = ['day', 'month', 'year']
date3.month = date3.month.apply(lambda x: x[:3])
date3.month = pd.to_datetime(date3.month, format='%b').dt.month.apply(str)
date3.day = date3.day.replace(np.nan, 1)


date4.columns = ['month', 'year']
date4['day'] = '1'

date5.columns = ['year']

date5['day'] = '1'
date5['month'] = '1'

date_all = pd.concat([date1, date2, date3, date4, date5])
date_all['month'] = date_all['month'].apply(str).apply(lambda x: x.lstrip('0'))
date_all['day'] = date_all['day'].apply(str).apply(lambda x: x.lstrip('0'))
date_all['date'] = date_all['month'] + '/' + date_all['day'] + '/' + date_all['year']
date_all['date'] = pd.to_datetime(date_all['date'], infer_datetime_format=True)
date_all = date_all.sort_index()
date_order = [item[0] for item in sorted(enumerate(date_all['date']), key=lambda x: x[1])]
print(date_order)