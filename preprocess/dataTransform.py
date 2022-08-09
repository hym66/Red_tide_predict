import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

datapath="../file"
oriname="/data_all.csv"
changedname="/data_all_transformed.csv"
data=pd.read_csv(datapath+oriname)

#data.drop(columns=['time', 'latitude', 'longtitude'], inplace=True)#这五个变量都不要
title=data.columns
#del data['time', 'latitude', 'longtitude', 'wind_speed', 'wind_stress']
#data.to_csv(datapath+changedname)
sheet1=pd.DataFrame(columns=title)
def getrange(data,level):
    max=0.0
    min=0.0
    for i in data:
       if i=='--':
            i=min
       if float(i) <=min:
            min=float(i)
       if float(i) >=max:
            max=float(i)
    print(min)
    singlepart=(max-min)/3
    lowmin=min+singlepart/4
    min+=singlepart
    medium=(min+lowmin)/2
    range=[lowmin,medium,min,max]
    return range

def mappingandsaving(data,i,sheet1):
    datacol=data[i].tolist()
    y = []
    for counting in range(len(datacol)):
        y.append(counting)
    #plt.scatter(datacol, y,s=0.05)
    #plt.savefig(datapath+"/"+i+".png")
    #plt.show()
    ranging=getrange(datacol,3)
    print(ranging,i)
    j=0
    for da in datacol:
        if np.isnan(da)==True:
           continue
        if da=='--':
            datacol[j]=1
        if da<=ranging[0]:
            datacol[j]=0
        elif da<=ranging[1]:
            datacol[j]=1
        elif da<=ranging[2]:
            datacol[j]=2
        else:
            datacol[j]=3
        #count=int(0)
        #for level in range:
         #   if da<level:
          #      datacol[j]=int(count)
           #     break        j+=1
            #count+=int(1)
    #print(i)
    sheet1[i]=datacol

def deletespace(dataf,num):
    dataforsave=pd.DataFrame(columns =dataf.columns)
    title=dataf.columns
    for i in title:
        datacol=dataf[i].tolist()
        datacolnew=[]
        if i=='time':
            for index,j in enumerate(datacol):
                if index%2==1 and index<num:
                    datacolnew.append(datacol[index])
                if index>=num:
                    break
            dataforsave[i] = datacolnew
            continue
        for index,j in enumerate(datacol):
            if np.isnan(j)==False and index<num:
                #print(j)
                datacolnew.append(j)
            if index >= num:
                break
        dataforsave[i]=datacolnew
    name="/data_all_conv.csv"
    dataforsave.to_csv(datapath+name,index=0)



deletespace(data,1000000)
# for i in title:
#     #if i!='nppv':
#         #continue
#     mappingandsaving(data,i,sheet1)
#
# sheet1.to_csv(datapath+changedname,index=0)


