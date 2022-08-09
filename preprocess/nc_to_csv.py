import netCDF4 as nc
import csv
import datetime as dt

def getDate(d):

    # if isinstance(d, dt.datetime):
    #     return d
    # if isinstance(d, cftime.DatetimeNoLeap):
    #     return dt.datetime(d.year, d.month, d.day, d.hour, d.minute, d.second)
    # elif isinstance(d, cftime.DatetimeGregorian):
    #     return dt.datetime(d.year, d.month, d.day)
    return d
    # elif isinstance(d, np.datetime64):
    #     return d.astype(dt.datetime)
    # else:
    #     raise Exception("Unknown value: {} type: {}".format(d, type(d)))


filename1 = r'..\file\data1.nc'
filename2 = r'..\file\data3.nc'
filename3 = r'..\file\data4.nc'

f1 = nc.Dataset(filename1)
f2 = nc.Dataset(filename2)
f3 = nc.Dataset(filename3)

data1 = f1.variables.keys()
data2 = f2.variables.keys()
data3 = f3.variables.keys()

time = nc.num2date(f3.variables['time'][:],'hours since 1950-1-1 00:00:00').data
depth = f3.variables['depth'][:]
latitude = f3.variables['latitude'][:]
longitude = f3.variables['longitude'][:]

vo = f1.variables['vo'][:]
uo = f1.variables['uo'][:]
thetao = f1.variables['thetao'][:]
zos = f1.variables['zos'][:]


kd = f2.variables['KD490'][:]

o2 = f3.variables['o2'][:]
chl = f3.variables['chl'][:]
no3 = f3.variables['no3'][:]
po4 = f3.variables['po4'][:]
si = f3.variables['si'][:]
nppv = f3.variables['nppv'][:]

print('开始计算数据总数')
tot = 0
ite = 0
for i in range(len(time)):
    for j in range(len(depth)):
        for k in range(len(latitude) - 1):
            for m in range(len(longitude) - 1):
                if (vo[i][j][k * 3][m * 3 - 3] != '--' and uo[i][j][k * 3][m * 3] != '--' and thetao[i][j][k * 3][m * 3] != '--'
                    and zos[i][k * 3][m * 3] != '--' and kd[i][k * 6][m * 6] != '--' and o2[i][j][k][m] != '--'
                    and chl[i][j][k][m] != '--' and no3[i][j][k][m] != '--' and po4[i][j][k][m] != '--'
                    and si[i][j][k][m] != '--' and nppv[i][j][k][m] != '--'):
                    tot+=1

print(tot)
print('数据总数计算完成，开始转换！')


with open('../file/data_train.csv',mode='w') as ice_file:
    ice_writer=csv.writer(ice_file,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
    ['time', 'latitude', 'longtitude', 'vo', 'uo', 'thetao', 'zos', 'kd', 'o2', 'chl', 'no3', 'po4', 'si', 'nppv'])#依次是列名，时间，纬度，经度及降雨

    for i in range(len(time)):
        if ite>0.7 * tot:
            break
        for j in range(len(depth)):
            if ite > 0.7 * tot:
                break
            for k in range(len(latitude)-1):
                if ite>0.7*tot:
                    break
                for m in range(len(longitude)-1):
                    if(vo[i][j][k*3][m*3-3]!='--' and uo[i][j][k*3][m*3]!='--'  and chl[i][j][k][m]!='--' and no3[i][j][k][m]!='--' and po4[i][j][k][m]!='--' and si[i][j][k][m]!='--' and nppv[i][j][k][m]!='--' and ):
                        ite+=1
                        if ite <= 0.7*tot:
                            ice_writer.writerow([getDate(time[i]),

                                                 latitude[k],
                                                 longitude[m],

                                                 vo[i][j][k * 3][m * 3 - 3],
                                                 uo[i][j][k * 3][m * 3],
                                                 thetao[i][j][k * 3][m * 3],
                                                 zos[i][k * 3][m * 3],

                                                 # wind_speed[i][k][m],
                                                 # wind_stress[i][k][m],

                                                 kd[i][k * 6][m * 6],

                                                 o2[i][j][k][m],
                                                 chl[i][j][k][m],
                                                 no3[i][j][k][m],
                                                 po4[i][j][k][m],
                                                 si[i][j][k][m],
                                                 nppv[i][j][k][m]
                            ])


print('训练集生成完毕！现在开始生成验证集...')

ite = 0
with open('../file/data_validate.csv', mode='w') as ice_file:
    ice_writer = csv.writer(ice_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    ice_writer.writerow(
        ['time', 'latitude', 'longtitude', 'vo', 'uo', 'thetao', 'zos', 'kd', 'o2', 'chl', 'no3', 'po4', 'si',
         'nppv'])  # 依次是列名，时间，纬度，经度及降雨

    for i in range(len(time)):
        for j in range(len(depth)):
            for k in range(len(latitude) - 1):
                for m in range(len(longitude) - 1):
                    if (vo[i][j][k * 3][m * 3 - 3] != '--' and uo[i][j][k * 3][m * 3] != '--' and thetao[i][j][k * 3][
                        m * 3] != '--' and zos[i][k * 3][m * 3] != '--' and kd[i][k * 6][m * 6] != '--' and o2[i][j][k][
                        m] != '--' and chl[i][j][k][m] != '--' and no3[i][j][k][m] != '--' and po4[i][j][k][
                        m] != '--' and si[i][j][k][m] != '--' and nppv[i][j][k][m] != '--'):
                        ite += 1
                        if (ite >= 0.7 * tot):
                            ice_writer.writerow([getDate(time[i]),

                                                 latitude[k],
                                                 longitude[m],

                                                 vo[i][j][k * 3][m * 3 - 3],
                                                 uo[i][j][k * 3][m * 3],
                                                 thetao[i][j][k * 3][m * 3],
                                                 zos[i][k * 3][m * 3],

                                                 # wind_speed[i][k][m],
                                                 # wind_stress[i][k][m],

                                                 kd[i][k * 6][m * 6],

                                                 o2[i][j][k][m],
                                                 chl[i][j][k][m],
                                                 no3[i][j][k][m],
                                                 po4[i][j][k][m],
                                                 si[i][j][k][m],
                                                 nppv[i][j][k][m]
                                                 ])
                        else:
                            continue

print('转换成功！')



