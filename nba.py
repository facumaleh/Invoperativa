from nba_api.stats.endpoints import commonplayerinfo
import pandas as pd
import picos

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
df = pd.read_csv (r'/Users/facundomaleh/Desktop/DatosNBAStat.csv')   #read the csv file (put 'r' before the path string to address any special characters in the path, such as '\'). Don't forget to put the file name at the end of the path + ".csv"

df.columns = df.columns.str.replace(' ', '')
print("\n\n", df)

ab= df.to_numpy()


        

##Generamos df para cada cosa
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
P = picos.Problem()
x = picos.BinaryVariable('x', len(ab))

#len(ab[0])

REBOTES= picos.Constant("PUNTOS", ab[:,-4])
PUNTOS= ab[:,-6]
ASISTENCIAS= ab[:,-5]

P.set_objective= 0.7* sum( PUNTOS*x)/5 +0.1* sum( REBOTES *x)/5 + 0.2* sum( ASISTENCIAS *x)/5 

P.add_constraint(sum(x)==5)

print("/////////////////////////////////////")
P.options.verbosity=0
print(P)
P.solve(solver= 'glpk')

print('x=', x)
print(P.value)

"""
