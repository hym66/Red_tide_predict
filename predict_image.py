import matplotlib.pyplot as plt###引入库包
import cartopy.crs as ccrs
import cartopy.feature as cfeature#预定义常量
import numpy as np
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter #导入Cartopy专门提供的经纬度的Formatter
import matplotlib as mpl
import pandas as pd
import csv
#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# x_extent = [ 0, 60, 120, 180, 240, 300, 360]#经纬度范围,#直接使用(-180,-120,-60,0,60,120,180)会异常，需要写成(0, 60, 120, 180, 240, 300, 360)的形式
# y_extent = [ -90,-60, -30, 0, 30, 60,90]

# x_extent = np.linspace(0,360,60)
# y_extent = np.linspace(-90,90,30)

x_extent = [118,120,122,124,126,128]
y_extent = [23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]

fig = plt.figure(figsize=(4, 4), dpi=200)
ax=plt.subplot(1,1,1,projection=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND)#添加陆地
ax.add_feature(cfeature.COASTLINE,lw = 0.3)#添加海岸线
ax.add_feature(cfeature.RIVERS,lw = 0.25)#添加河流
ax.add_feature(cfeature.LAKES)#指定湖泊颜色为红色#添加湖泊
#ax.add_feature(cfeature.BORDERS, linestyle = '-',lw = 0.25)#不推荐，因为该默认参数会使得我国部分领土丢失
ax.add_feature(cfeature.OCEAN)#添加海洋

ax.set_xticks(x_extent, crs=ccrs.PlateCarree())#添加经纬度
ax.set_yticks(y_extent, crs=ccrs.PlateCarree())

# 利用Formatter格式化刻度标签
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)

ax.set_extent([118,127,23,37],crs = ccrs.PlateCarree()) #选取经度为0°E-180°E，纬度为0°N-90°N的区域

ax.set_title('东海2015.9.15赤潮叶绿素含量预测') # 添加标题

x = []
y = []
i = 25.00
j = 120.00
while i < 35.00:
    y.append(i)
    i += 0.25
while j < 125.00:
    x.append(j)
    j += 0.25
X,Y = np.meshgrid(x,y)
data = pd.read_csv("./file/2015-9-15.csv")

result = []                  # 创建一个空列表
for i in range(40):      # 创建一个5行的列表（行）
    result.append([])        # 在空的列表中添加空的列表
    for j in range(20):  # 循环每一行的每一个元素（列）
        result[i].append(0)  # 为内层列表添加元素

for i in range(40):
    for j in range(20):
        lat = 25.00 + i*0.25
        lon = 120.00 + j*0.25
        for k in range(len(data)):
            if(data['latitude'][k]==lat and data['longitude'][k]==lon):
                result[i][j] = data['chl'][k]


plt.contourf(X,Y,result,100,alpha=0.5,cmap=plt.cm.hot)
# plt.xticks(())
# plt.yticks(())



plt.tick_params(labelsize=5)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

plt.show()
