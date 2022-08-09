import numpy as np
import pandas as pd

dataset = pd.read_csv('../file/data_all_conv.csv', encoding="gbk", header=0,
                      infer_datetime_format=True, engine='c'
                      )
lat = []
lon = []
i = 25.00
j = 120.00
while i < 35.00:
    lat.append(i)
    i += 0.25
while j < 125.00:
    lon.append(j)
    j += 0.25

#print(dataset.columns)
df = dataset.iloc[:, 3:16].fillna(0)
time = dataset.iloc[:, 0:3]

dataset_t = pd.DataFrame(df)
#print(dataset_t.columns)
X_value = dataset_t.iloc[:, 3:]
y_value = dataset_t.iloc[:, 3]
#print(y_value)
# # # Normalized the data
# X_scaler = MinMaxScaler(feature_range=(-1, 1))
# y_scaler = MinMaxScaler(feature_range=(-1, 1))
# X_scaler.fit(X_value)
# y_scaler.fit(y_value)
#
# X_scale_dataset = pd.DataFrame(X_scaler.fit_transform(X_value))
# y_scale_dataset = pd.DataFrame(y_scaler.fit_transform(y_value))
# X_scaler_dataset=X_scale_dataset.fillna(0)
# dump(X_scaler, open('X_scaler.pkl', 'wb'))
# dump(y_scaler, open('y_scaler.pkl', 'wb'))
dataset_t = pd.concat([time, dataset_t], axis=1)
data = dataset_t.groupby('time')
time_index = []
# 得到一个timelist
days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
name = 'Conv-LSTM'
time_list = []
count = 0
for daily_data in data:
    count = count + 1
    print(count)
    #print(len(lat))
    pic = np.zeros([6, 40, 20])  # 宽 长 高 time lat lon
    dd = pd.DataFrame(daily_data[1])
    temp = 0
    for index in range(dd.shape[0]):
        time_index.append(daily_data[0])
        for k in range(0, 6):
            for i in range(len(lat)):
                if dd.iloc[index]['latitude'] == lat[i]:
                    break
            for j in range(len(lon)):
                if dd.iloc[index]['longtitude'] == lon[j]:
                    break
            #print(j)
            #print(i)
            if k == 0:
                pic[k][i][j] = (dd.iloc[index][k + 3]) * 150
            else:
                pic[k][i][j] = dd.iloc[index][k + 3]

    time_list.append(pic)
dataset = np.array(time_list)
np.save("../file/time_list1.npy", dataset)