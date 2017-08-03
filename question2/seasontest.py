from season import ScoreBoard, Season
from readwrite import read_teams, read_games
import datetime

teamlist = read_teams('Analytics_Attachment.xlsx')
#teamlist.print()
gamelist = read_games('Analytics_Attachment.xlsx')
#gamelist.print() 

#sb = ScoreBoard(teamlist, gamelist, gamelist.gamelist[len(gamelist.gamelist)].date)

#teamnamelist = list()
#for ele in teamlist.teamlist:
#    if teamlist.teamlist[ele].confiname == 'East':
#        teamnamelist.append(teamlist.teamlist[ele].name)

#print(sb.get_win_lose_remain_total('Cleveland Cavaliers', teamnamelist))

#print(sb.get_div_list('Boston Celtics'))

#conf_list = sb.get_conf_list('Boston Celtics')
#conf_list.sort()
#print(conf_list)

#li = sb.sort_by_lose(teamnamelist)
#print(li)

#print(sb.must_div_lead('Orlando Magic'))


Se = Season(teamlist, gamelist)
Se.run()
'''
d1=datetime.datetime.strptime('3/20/2017','%m/%d/%Y')

sb = ScoreBoard(teamlist, gamelist, d1)
sb.win_all('Minnesota Timberwolves')
sb.win_all('Golden State Warriors')
sb.win_all('San Antonio Spurs')
sb.win_all('LA Clippers')
sb.win_all('Houston Rockets')
sb.win_all('Memphis Grizzlies')
sb.win_all('Oklahoma City Thunder')
sb.win_all('Utah Jazz')

teamnamelist = list()
for ele in teamlist.teamlist:
    if teamlist.teamlist[ele].confiname == 'West':
        teamnamelist.append(teamlist.teamlist[ele].name)
t,s = sb.sort_by_lose(teamnamelist)
for i in range(len(t)):
    print(t[i],s[i])
'''
