# file name is season
import datetime
import copy


class Team(object):
    def __init__(self, teamname, diviname, confiname):
        self.name = teamname
        self.diviname = diviname
        self.confiname = confiname


class TeamsList(object):

    def __init__(self):
        self.teamlist = dict()

    def add(self, teamname, diviname, confiname):
        newteam = Team(teamname, diviname, confiname)
        self.teamlist[newteam.name] = newteam
    
    def print(self):
        for team in self.teamlist:
            t = self.teamlist[team]
            print(t.name, t.diviname, t.confiname)


class Game(object):

    def __init__(self, date, home, away, win):
        self.date = date
        self.home = home
        self.away = away
        if win == 'Home':
            self.win = self.home
        else:
            self.win = self.away 


class GamesList(object):

    def __init__(self):
        self.gamelist = dict()
        self.gameno = 0
   
    def __len__(self):
        return self.gameno

    def add(self, date, home, away, win):
        newgame = Game(date, home, away, win)
        self.gameno += 1
        self.gamelist[self.gameno] = newgame

    def print(self):
        for i in range(1,self.gameno+1):
            g = self.gamelist[i]
            print(g.date, g.home, g.away, g.win, i)    


class ScoreBoard(object):
    
    def __init__(self, teamlist, gamelist, date):
        self.teamlist = teamlist.teamlist
        self.gamelist = gamelist.gamelist
        self.lenth = len(gamelist)
        for i in range(1, self.lenth+1):
            if self.gamelist[i].date > date:
                self.gamelist[i].win = ''
        self.totalnamelist = list()
        for ele in self.teamlist:
            self.totalnamelist.append(self.teamlist[ele].name)

    def print(self, teamnow):
        for i in range(1, self.lenth+1):
            if (self.gamelist[i].home == teamnow) or (self.gamelist[i].away == teamnow):
                print(self.gamelist[i].home, self.gamelist[i].away, self.gamelist[i].win)

    def get_win_lose_remain_total(self, teamname, teamnamelist):
        wins = 0
        lose = 0
        remain = 0
        total = 0
        for i in range(1, self.lenth+1):
            if ((self.gamelist[i].home == teamname) or 
                (self.gamelist[i].away == teamname)):
                flag = False
                for othername in teamnamelist:
                    if (((othername == self.gamelist[i].home) or
                        (othername == self.gamelist[i].away)) and 
                        (othername != teamname)):
                        flag = True
                        break
                if flag:
                    total += 1
                    if self.gamelist[i].win == teamname:
                        wins += 1
                    else:
                        if self.gamelist[i].win == '':
                            remain += 1
        return wins, total-wins-remain, remain, total

    def get_lose(self, teamname, teamnamelist):
        win, lose, remain, total = self.get_win_lose_remain_total(teamname, teamnamelist)
        return lose

    def sort_by_lose(self, teamnamelist):
        totalteamlist = list(self.totalnamelist)
        for i in reversed(range(0,len(teamnamelist))):
            for j in range(0,i):
                if (self.get_lose(teamnamelist[j], totalteamlist) > 
                    self.get_lose(teamnamelist[j+1], totalteamlist)):
                    temp = teamnamelist[j]
                    teamnamelist[j] = teamnamelist[j+1]
                    teamnamelist[j+1] = temp
        scorelist = list()
        for i in range(0, len(teamnamelist)):
            win, lose, remain, total = self.get_win_lose_remain_total(teamnamelist[i], totalteamlist)
            scorelist.append(lose)
        return teamnamelist, scorelist

    def win_all(self, teamname):
        for i in range(1, self.lenth+1):
            if (((self.gamelist[i].home == teamname) or (self.gamelist[i].away == teamname)) and
                (self.gamelist[i].win == '')):
                self.gamelist[i].win = teamname

    def lose_to_list(self, tlist, target):
        for i in range(1, self.lenth+1):
            g = self.gamelist[i]
            if  (g.win == ''):
                if (g.home in tlist) and (g.away in target):
                    g.win = g.away
                if (g.away in tlist) and (g.home in target):
                    g.win = g.home

    def get_remain_games_list(self, conflist):
        remainslist = list()
        namelist = list()
        for i in range(1, self.lenth+1):
            g = self.gamelist[i]
            if  (g.win == '') and (g.home in conflist) and (g.away in conflist):
                remainslist.append(i)
                if g.home not in namelist:
                    namelist.append(g.home)
                if g.away not in namelist:
                    namelist.append(g.away)
        return remainslist, namelist

    def f_in_list(self, teamnow, tlist, gglist):
        # 1 first in list
        # 0 tie first in list
        # 2 the other
        ttlist = list(tlist)
        for i in reversed(range(0,len(ttlist))):
            for j in range(0,i):
                if (self.get_lose(ttlist[j], gglist) > 
                    self.get_lose(ttlist[j+1], gglist)):
                    temp = ttlist[j]
                    ttlist[j] = ttlist[j+1]
                    ttlist[j+1] = temp
        scorelist = list()
        for i in range(0, len(ttlist)):
            win, lose, remain, total = self.get_win_lose_remain_total(ttlist[i], gglist)
            scorelist.append(lose)
        for i in range(0, len(ttlist)):
            if ttlist[i] == teamnow:
                position = i
        if (position == 0) and (scorelist[0] != scorelist[1]):
            return 1
        if scorelist[position] > scorelist[0]:
            return 2
        return 0


class Season(object):
    
    def __init__(self, teamlist, gamelist):
        self.teamslist = teamlist
        self.wholegame = gamelist
        self.date = gamelist.gamelist[1].date
        self.outlist = list()
        self.outlistdate = dict()
        self.totalnamelist = list()
        for ele in self.teamslist.teamlist:
            self.totalnamelist.append(self.teamslist.teamlist[ele].name)
                        
    def get_divi_list(self, teamname):
        div = self.teamslist.teamlist[teamname].diviname
        divlist = list()
        for othername in self.teamslist.teamlist:
            if self.teamslist.teamlist[othername].diviname == div:
                divlist.append(othername)
        return divlist

    def get_conf_list(self, teamname):
        conf = self.teamslist.teamlist[teamname].confiname
        confilist = list()
        for othername in self.teamslist.teamlist:
            if self.teamslist.teamlist[othername].confiname == conf:
                confilist.append(othername)
        return confilist

    def get_other_conf_list(self, teamname):
        conf = self.teamslist.teamlist[teamname].confiname
        if conf == 'West':
            conf = 'East'
        else:
            conf = 'West'
        confilist = list()
        for othername in self.teamslist.teamlist:
            if self.teamslist.teamlist[othername].confiname == conf:
                confilist.append(othername)
        return confilist

    def run(self):
        '''
        from the first date of the gamelist to the end.
        get the last team which is not in the outlist of one conference.
        see if it has a hope to play in playoffs.
        '''    

        def hope(teamnow, sbn):
            '''
            see if a team still have a chance to play in the playoffs.
            '''

            def step1(i, times, teamnow, sbn):
                '''
                let the frist 7 teams win
                '''

                conflist = self.get_conf_list(teamnow)
                listscore, scores = sbn.sort_by_lose(conflist)
                if times<7:
                    topi = i
                    if listscore[topi] == teamnow:
                        topi += 1
                        i += 1
                    while scores[i] == scores[i+1]:
                        i += 1
                        if i+1 == len(scores):
                            break
                    position = 0
                    for ii in range(0, len(listscore)):
                        if listscore[ii] == teamnow:
                            position = ii
                    if position < 7:
                        #print('1',teamnow)
                        return True
                    for j in range(topi, i+1):
                        if listscore[j] != teamnow:
                            #swicth (topi, j)
                            temp = listscore[j]
                            listscore[j] = listscore[topi]
                            listscore[topi] = temp
                            tmp = scores[j]
                            scores[j] = scores[topi]
                            scores[topi] = tmp
                            sbn2 = copy.deepcopy(sbn)
                            sbn2.win_all(listscore[topi])
                            if step1(topi+1, times+1, teamnow, sbn2):
                                return True
                    return False
                else:
                    # all teams lose to other side
                    otherconf = self.get_other_conf_list(teamnow)
                    sbn.lose_to_list(conflist, otherconf)
                    # remaining games
                    glist, tlist = sbn.get_remain_games_list(conflist)
                    # average process step2(remains, teamnow, sbn, teamlist, gamelist)
                    remains = len(glist)
                    listscore, scores = sbn.sort_by_lose(conflist)
                    position = 0
                    for ii in range(0, len(listscore)):
                        if listscore[ii] == teamnow:
                            position = ii
                    if position < 7:
                        #print('2',teamnow)
                        return True
                    while remains != 0:
                        remains = step2(remains, sbn, tlist, glist)
                    listscore, scores = sbn.sort_by_lose(conflist)
                    position = 0
                    for ii in range(0, len(listscore)):
                        if listscore[ii] == teamnow:
                            position = ii
                            if ii+1 < len(listscore):
                                if scores[ii] == scores[ii+1]:
                                    #swicth ii, ii+1
                                    temp = listscore[ii]
                                    listscore[ii] = listscore[ii+1]
                                    listscore[ii+1] = temp
                                    tmp = scores[ii]
                                    scores[ii] = scores[ii+1]
                                    scores[ii+1] = tmp
                                else:
                                    break
                    if position <= 7:
                        #print('3',teamnow)                     
                        return True
                    else:
                        if scores[position] > scores[7]:
                            return False
                        else:
                            tielist = list()
                            for ii in range(0,len(listscore)):
                                if scores[ii] == scores[position]:
                                    tielist.append(listscore[ii])
                            return tie_d(teamnow, tielist, sbn)

            def step2(remains, sbn, teamlist, gamelist):
                '''
                make the remains stay in average
                '''

                leavelist, _ = sbn.sort_by_lose(teamlist)
                high = leavelist[0]
                low = leavelist[-1]
                for high in range(0, len(leavelist)):
                    for low in reversed(range(high, len(leavelist))):
                        for i in gamelist:
                            if (((sbn.gamelist[i].home == high) or (sbn.gamelist[i].away == high)) and
                                ((sbn.gamelist[i].home == low) or (sbn.gamelist[i].away == low))):
                                sbn.gamelist[i].win = low
                                gamelist.remove(i)
                                return remains-1
                return 0

            def tie_d(teamnow, tielist, sbn):
                '''
                decided a tie
                '''
                
                def in_same_divi(tielist, sbn):
                    diviname = sbn.teamlist[tielist[0]].diviname
                    for t in tielist:
                        if sbn.teamlist[t].diviname != diviname:
                            return False
                    return True

                def division_leader_win(teamnow, tielist, sbn):
                    # 1 im leader
                    # 2 other is leader and im not
                    # 0 the other
                    teamnowdivi = self.get_divi_list(teamnow)
                    tlist = list()
                    for ele in self.teamslist.teamlist:
                        tlist.append(self.teamslist.teamlist[ele])
                    flag = sbn.f_in_list(teamnow, teamnowdivi, tlist)
                    if flag == 2:
                        for team in tielist:
                            teamdivi = self.get_divi_list(team)
                            if sbn.f_in_list(team, teamdivi, tlist) != 2:
                                return 2
                        return 0
                    else:
                        for team in tielist:
                            teamdivi = self.get_divi_list(team)
                            if sbn.f_in_list(team, teamdivi, tlist) != 2:
                                return 0
                        return 1

                if len(tielist) == 2:
                    # 1
                    flag = sbn.f_in_list(teamnow, tielist, tielist)
                    if flag == 1:
                        return True
                    if flag == 2:
                        return False
                    # 2
                    if division_leader_win(teamnow, tielist, sbn) == 1:
                        return True
                    else:
                        if division_leader_win(teamnow, tielist, sbn) == 2:
                            return False
                    # 3
                    if in_same_divi(tielist, sbn):
                        # if teamnow is high: return True else: return False
                        divilist = self.get_divi_list(tielist[0])
                        flag = sbn.f_in_list(teamnow, tielist, divilist)
                        if flag == 1:
                            return True
                        if flag == 2:
                            return False                        
                    # 4
                    conflist = self.get_conf_list(tielist[0])
                    flag = sbn.f_in_list(teamnow, tielist, conflist)
                    if flag == 1:
                        return True
                    if flag == 2:
                        return False
                    # 5
                    playofflist, _ = sbn.sort_by_lose(conflist)
                    playofflist = playofflist[0:7]
                    for ele in tielist:
                        if ele not in playofflist:
                            playofflist.append(ele)
                    flag = sbn.f_in_list(teamnow, tielist, playofflist)
                    if flag == 1:
                        return True
                    if flag == 2:
                        return False


                else:
                    # 1
                    if division_leader_win(teamnow, tielist, sbn) == 1:
                        return True
                    else:
                        if division_leader_win(teamnow, tielist, sbn) == 2:
                            return False
                    # 2
                    flag = sbn.f_in_list(teamnow, tielist, tielist)
                    if flag == 1:
                        return True
                    if flag == 2:
                        return False
                    # 3
                    if in_same_divi(tielist, sbn):
                        # if teamnow is high: return True else: return False
                        divilist = self.get_divi_list(tielist[0])
                        flag = sbn.f_in_list(teamnow, tielist, divilist)
                        if flag == 1:
                            return True
                        if flag == 2:
                            return False   
                    # 4
                    conflist = self.get_conf_list(tielist[0])
                    flag = sbn.f_in_list(teamnow, tielist, conflist)
                    if flag == 1:
                        return True
                    if flag == 2:
                        return False
                    # 5
                    playofflist, _ = sbn.sort_by_lose(conflist)
                    playofflist = playofflist[0:7]
                    for ele in tielist:
                        if ele not in playofflist:
                            playofflist.append(ele)
                    flag = sbn.f_in_list(teamnow, tielist, playofflist)
                    if flag == 1:
                        return True
                    if flag == 2:
                        return False                       
                return True

            sbn.win_all(teamnow)
            times = 0
            i = 0
            return step1(i, times, teamnow, sbn)
            
        def find_last(scorelist, scores, outlist):
            last = list()
            flag = True
            for i in reversed(range(len(scorelist))):
                if scores[i] <= 21:
                    break
                if not flag:
                    if (scores[i] == scores[i+1]) and (scorelist[i] not in outlist):
                        last.append(scorelist[i])
                    else:
                        break
                if i>8:
                    if scorelist[i] not in outlist:
                        last.append(scorelist[i])
                if flag and i == 8:
                    if scorelist[i] not in outlist:
                        last.append(scorelist[i])
                        flag = False
            return last
        
        now = self.date
        end = self.wholegame.gamelist[len(self.wholegame.gamelist)].date
        eastlist = self.get_conf_list('Boston Celtics')
        westlist = self.get_conf_list('Golden State Warriors')
        ######
        #now = datetime.datetime.strptime('3/16/2017','%m/%d/%Y')
        while now <= end:
            ###########################
            #print(now)
            #print(self.outlist)
            nowgame = copy.deepcopy(self.wholegame)
            sb = ScoreBoard(self.teamslist, nowgame, now)
            wscorelist, wscores = sb.sort_by_lose(westlist)
            escorelist, escores = sb.sort_by_lose(eastlist)
            wlast = find_last(wscorelist, wscores, self.outlist)
            elast = find_last(escorelist, escores, self.outlist)
            if wlast != []:
                for last in wlast:
                    sbn = copy.deepcopy(sb)
                    if not hope(last, sbn):
                        self.outlist.append(last)
                        self.outlistdate[last] = datetime.datetime.strftime(now, '%m/%d/%Y')
            if elast != []:
                for last in elast:
                    sbn = copy.deepcopy(sb)
                    if not hope(last, sbn):
                        self.outlist.append(last)
                        self.outlistdate[last] = datetime.datetime.strftime(now, '%m/%d/%Y')
            now = now + datetime.timedelta(days = 1)
        for teamname in self.totalnamelist:
            if teamname not in self.outlist:
                self.outlist.append(teamname)
                self.outlistdate[teamname] = 'Playoff'

    def get_outlist(self):
        return self.outlistdate
    
if __name__ == '__main__':
    team = TeamsList()
    game = GamesList()
    season = Season(team, game)
    season.run()
    season.get_outlist()
