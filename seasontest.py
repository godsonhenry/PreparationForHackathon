from season import ScoreBoard, Season
from readwrite import read_teams, read_games

teamlist = read_teams('Analytics_Attachment.xlsx')
#teamlist.print()
gamelist = read_games('Analytics_Attachment.xlsx')
#gamelist.print() 
sb = ScoreBoard(teamlist, gamelist, gamelist.gamelist[len(gamelist.gamelist)].date)

teamnamelist = list()
for ele in teamlist.teamlist:
    if teamlist.teamlist[ele].confiname == 'East':
        teamnamelist.append(teamlist.teamlist[ele].name)

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



