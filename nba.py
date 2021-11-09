from nba_api.stats.endpoints import commonplayerinfo
import pandas as pd
import picos
import numpy as np

"""
El obj del trabajo es encontrar el mejor equipo de la temporada, 
el mejor equipo de rookies y el mejor equipo de extranjeros de la temporada  
2019/2020 de la nba mediante la optimizaacion 
de nuestra funcion objetivo.

La funcion objetivo y restricciones fue fijata por experimentada en este deporte y sus gustos.
En este caso vamosa maximizar puntos rebotes y asistencias ponderandolos en 0.87 0.1 y 0.2 respectivamente.
Para tener un equipo balanceado se quieren tener 1 guards, 3fowards y 1 center.
Este equipo es ideal para el ritmo de juego de la nba ya que es un deporte muy dinamico 
y se necesita velocidad. Se quiere 1 bases para poder tner un buen movimiento de balon, 
3 fowards para generar situaciones de gol y un center  para bajar los rebotes.
Ademas se queire que el promedio de gol este por arriba de 23, 
el promedio de rebotes  sea por lo menos de 5 y el promedio de asistenia sea por lo menos 3.


"""

"""
headers = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:61.0) Gecko/20100101 Firefox/61.0',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.5',
    'Referer': 'https://stats.nba.com/',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
}
custom_headers = {
    'Host': 'stats.nba.com',
    'Connection': 'keep-alive',
    'Cache-Control': 'max-age=0',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
    'Accept-Encoding': 'gzip, deflate, br',
    'Referer': 'https://stats.nba.com/',
    'Accept-Language': 'en-US,en;q=0.9',
}
# Only available after v1.1.0
# Proxy Support, Custom Headers Support, Timeout Support (in seconds)
player_info = commonplayerinfo.CommonPlayerInfo(player_id=2544, headers=custom_headers, timeout=10)
#player_info = commonplayerinfo.CommonPlayerInfo(player_id=2544, headers=headers, timeout=100)
from nba_api.stats.static import players
players_info = players.get_players()
dfPlayers = pd.DataFrame(players_info)
L=[]
for k in dfPlayers.iloc:
    if k['is_active']:
        player_info = commonplayerinfo.CommonPlayerInfo(player_id=k['id'], headers=custom_headers, timeout=10)
        L.append(pd.concat([player_info.get_data_frames()[0],player_info.get_data_frames()[1],player_info.get_data_frames()[2]],axis=1,join='inner'))
dfStat=pd.concat(L)


"""
#df = pd.read_csv (r'/Users/facundomaleh/Desktop/DatosNBAStat.csv')   #read the csv file (put 'r' before the path string to address any special characters in the path, such as '\'). Don't forget to put the file name at the end of the path + ".csv"
df = pd.read_csv ('DatosNBAStat.csv')
df.columns = df.columns.str.replace(' ', '')
#print("\n\n", df)
ab= df.to_numpy()

        

"""##Generamos df para cada cosa
UsaDf= pd.DataFrame(df[df.COUNTRY == 'USA'])
ExtranjerosDF= pd.DataFrame(df[df.COUNTRY != 'USA'])
RookiesDf= pd.DataFrame(df[df.SEASON_EXP == 1])
VeteranosDf= pd.DataFrame(df[df.SEASON_EXP > 5])
CenterDf=pd.DataFrame(df[df.POSITION == 'Center'])
GuardDf=pd.DataFrame(df[df.POSITION == 'Guard'])
Forward_GuardDF=pd.DataFrame(df[df.POSITION == 'Forward-Guard'])
Center_FowardDf=pd.DataFrame(df[df.POSITION == 'Center-Forward'])
Foward_centerDf=pd.DataFrame(df[df.POSITION == 'Forward-Center'])
CenterFowardPostaDF = pd.concat([Center_FowardDf, Foward_centerDf], axis=0)

#RookieUsaDf= pd.DataFrame(df[RookiesDf.COUNTRY == 'USA'])


###De df a Numpy
USA= UsaDf.to_numpy()
Extranjeros= ExtranjerosDF.to_numpy()
Rookies= RookiesDf.to_numpy()
Veteranos= VeteranosDf.to_numpy()
Center= CenterDf.to_numpy()
Guard= GuardDf.to_numpy()
Forward_Guard= Forward_GuardDF.to_numpy()
Center_Foward= CenterFowardPostaDF.to_numpy()
"""


vg=[]

for i in range(len(df.POSITION)):

    if df.POSITION[i] == "Guard" or df.POSITION[i] =="Guard-Forward":
        vg.append(1)
    else:
        vg.append(0)
        
vf=[]

for i in range(len(df.POSITION)):

    if df.POSITION[i] == "Center-Forward" or df.POSITION[i] =="Guard-Forward" or df.POSITION[i] =="Forward" :
        vf.append(1)
    else:
        vf.append(0)


vc=[]

for i in range(len(df.POSITION)):

    if df.POSITION[i] == "Center" or df.POSITION[i] =="Forward-Center":
        vc.append(1)
    else:
        vc.append(0)


        

#

P = picos.Problem()
x = picos.BinaryVariable('x', len(ab))

REBOTES= picos.Constant("REBOTES", list(ab[:,-4]))
PUNTOS= picos.Constant("PUNTOS", list(ab[:,-6]))
ASISTENCIAS= picos.Constant("ASISTENCIAS", list(ab[:,-5]))

#PUNTOS= list(ab[:,-6])
#ASISTENCIAS= list(ab[:,-5])

#P.set_objective= 0.7* sum( PUNTOS.T*x)/5 
a=0.80
b=0.10
c=0.10
P.set_objective= a *sum( PUNTOS.T*x)/5 +b* sum( REBOTES.T*x)/5 + c* sum( ASISTENCIAS.T*x)/5 


##Quiero 1 guards
P.add_constraint(sum(vg*x)==1)
##Quiero 3 fowards
P.add_constraint(sum(vf*x)==2)
##Quiero 1 center
P.add_constraint(sum(vc*x)==2)

P.add_constraint(sum(x)==5)
##promedio minimo de 9 rebotes
P.add_constraint(sum(REBOTES.T*x)/5>=9)
##promedio minimo de 27 puntos
P.add_constraint(sum(PUNTOS.T*x)/5>=27)
##promedio minimo de 6 asistencias
P.add_constraint(sum(ASISTENCIAS.T*x)/5>=6)




print("/////////////////////////////////////")
P.options.verbosity=1
#print(P)
P.solve(solver= 'glpk')


#print(0.7* sum( PUNTOS.T*x)/5 +0.1* sum( REBOTES.T*x)/5 + 0.2* sum( ASISTENCIAS.T*x)/5)

indices = []
for i, v in enumerate(x):
    if v.value == 1:
        indices.append(i)
        
print(indices)

eqp=[]
for i in indices:
    print(i)
    eqp.append(ab[i,:])
print(eqp)

