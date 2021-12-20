import numpy as np
import matplotlib.pyplot as plt

def data(filename):
    f= open(filename,'r')
    data=[]
    i=1
    for line in f.readlines():
        newline=line.split()
        newline[0]=i
        i+=1
        for index in range(1,len(newline)):
            if newline[index]=='-':
                newline[index]=0
            elif len(newline[index]) >=5 and len(newline[index]) <= 7:
                a,b=map(int,newline[index].split(','))
                newline[index]=a*1000+b
            elif len(newline[index]) >=8 :
                a,b,c=map(int,newline[index].split(','))
                newline[index]=a*1000000+b*1000+c
            else:
                newline[index]=int(newline[index])
        data.append(newline)
    return data
def cof(x):
        return ((2*np.pi*x)/(((((x)/201)//1))*200))
pcr=np.array(data('pcr.txt'))
#pcr 날짜 전체확진자 국내발생 해외유입 사망자
vaccine=np.array(data('vaccine.txt'))
#vaccine 날짜 전체1차 전체2차 전체3차 AZ1 AZ2 F1 F2 Y1 M1 M2 F3 M3 Y3
lmp='local maximum point'
#전체 확진자 추이
x=np.linspace(pcr.T[0].min(),pcr.T[0].max(),len(pcr.T[0]))
y=pcr.T[1]
plt.xlabel('day after debut of covid19')
plt.ylabel('number of positive')
plt.plot(x,y,'.')
plt.plot(x[220],y[220],'o','red',label=lmp)
plt.plot(x[340],y[340],'o','red',label=lmp)
plt.plot(x[569],y[569],'o','red',label=lmp)
plt.title("positive scatter for whole range")
plt.legend()
plt.show()

#전체 사망자 추이
plt.title("death rate for whole range")
plt.ylabel('death rate')
plt.xlabel('day after debut of covid19')
plt.plot(x,pcr.T[4]/pcr.T[1],'-')
plt.show()

#위드코로나 시행 전 코로나 확진자 추이
plt.plot(x[:len(x)-50],y[:len(y)-50],'.')
plt.plot(x[220],y[220],'o','red') 
plt.plot(x[340],y[340],'o','red') #120
plt.plot(x[569],y[569],'o','red') #220
plt.xlabel('day after debut of covid and before WithCorona')
plt.ylabel('number of positive')
plt.title("positive rate before WithCorona")
plt.show()

# 위드코로나 전 내가 예측한 라인 피팅
x1=x[200:len(x)-50]
y2=y[200:len(y)-50]
poly_fit=np.polyfit(x1,y2,7)
poly_1d=np.poly1d(poly_fit)
xs=np.linspace(x1.min(),x1.max()+50) #오픈소스를 활용한 라인 피팅
ys=poly_1d(xs) 
plt.plot(xs,np.abs(ys),'k-',label='line pitting by polyfit(opensource)')
y1=np.zeros(x1.shape)
i=0
for node in x1:
    y1[i]=np.abs((np.cos(cof(node)))*((((np.abs(node)/200))//1)))
    i+=1
plt.title('predicting pcr positives with line pitting')
plt.plot(600,'-')
coefs=np.vstack((x1,y1,np.ones(x1.shape)))
coefs=np.matmul(np.linalg.pinv(coefs.T),y[:len(x1)])
def pcrline(x,liss):
    return 2*liss[0]*x +8*liss[1]*(np.sin(cof(x1))*((((np.abs(x1)/200))//1)**2))+liss[2]
x3=x[len(x)-50:]
y3=y[len(y)-50:]
plt.plot(x3,y3,'y.',alpha=1,label='actual pcr positive after WithCorona')
ploy_fit1=np.polyfit(x3,y3,1)
poly_1d=np.poly1d(ploy_fit1)
xs1=np.linspace(x3.min(), x3.max())
ys1=poly_1d(xs1)
plt.plot(xs1,ys1,'y-',label='line pitting after withCorona')
plt.plot(x1,pcrline(x1,coefs),'r-',label='line pitting by myself')
plt.plot(x[:len(x)-50],y[:len(y)-50],'b.',alpha=0.3,label='actual pcr positive')
plt.plot(x[220],y[220],'o','red')
plt.plot(x[340],y[340],'o','red')
plt.plot(x[569],y[569],'o','red')
plt.annotate('local max',xy=(x[220],y[220]),arrowprops=dict(facecolor='black',shrink=0.0005,alpha=0.7))
plt.annotate('local max',xy=(x[340],y[340]),arrowprops=dict(facecolor='black',shrink=0.0005,alpha=0.7))
plt.annotate('local max',xy=(x[569],y[569]),arrowprops=dict(facecolor='black',shrink=0.0005,alpha=0.7))
plt.xlabel('day after debut of covid and before WithCorona')
plt.ylabel('number of positive')
plt.legend()
plt.show()
# 내가 라인 피팅에 사용한 함수 
plt.title('the sine wave used self line pitting')
x4=np.linspace(200,100000)
plt.plot(x4,100*np.cos(cof(x4)),'k-')
plt.legend()
plt.show()
# 백신과 사망률 관계 추이
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
pop=51821669 #대한민국 총 인구 FOR 백신 접종률
data={
      'deathRate':((pcr[403:len(pcr)-2,4]/pcr[403:len(pcr)-2,1])*10e6)//1,
      'vaccine':vaccine[:,1]/pop,
      'AZ':vaccine[:,5]/pop,
      'Fizer':vaccine[:,7]/pop, #데이터에 쓰일 확진자,총 백신 접종상황, 백신별 접종 현황
      'Y':vaccine[:,8]/pop,
      'Modern':vaccine[:,10]/pop
      }
data=pd.Series(data)
x=np.array(data['vaccine'])
y=np.array(data['deathRate'])
#회귀 분석 툴을 이용한 4차원 회귀본석 파트
poly_fit=np.polyfit(x,y,4)
poly_1d=np.poly1d(poly_fit)
xs=np.linspace(x.min(),x.max()) #현재 나와있는 데이터 이후의 예측 폴리핏
ys=poly_1d(xs)
plt.title('deathRate with vaccination rate')
plt.scatter(x,y, label='actual deathRate')
plt.plot(xs,ys,color='red',label='line pitting by poly_fit')
formular = 'deathRate ~ vaccine'
result=smf.ols(formular,data).fit() #statsmodels를 이용한 선형 분석
print('백신 접종률과 사망률에 대한 분석','\n',result.summary()) 
xs1=np.linspace(xs.min(),xs.max())
ys1=6.23e-05*xs1+5.296e+4
plt.plot(xs1,ys1,'green',label='regression by overall vaccine') #1차원에는 잘 맞지 않는다.

formula2='deathRate~AZ+Fizer+Y+Modern' #백신별로 변수를 만들어 학습
result2=smf.ols(formula2,data).fit() #학습
def deathForVaccine(A,F,Y,M):
    return 7.365e+04 + 3.923e+04*A + 9.816e+04*F - 2.431e+06*Y + 3.382e+05*M
plt.plot(x,deathForVaccine(data['AZ'],data['Fizer'],data['Y'],data['Modern']),'k-',label='regression by sum of each vaccine')
plt.xlabel('vaccination rate')
plt.ylabel('death rate')
plt.legend()
plt.show()
print('백신별 사망률에 대한 분석','\n',result2.summary())
I=np.eye(4)
for i in range(4):
    now=x[-1]*I[i]
    print(deathForVaccine(now[0],now[1],now[2],now[3]))
