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

    #def get_divi_list(self, teamname):
    #    div = self.teamlist[teamname].diviname
    #    divlist = list()
    #    for othername in self.teamlist:
    #        if self.teamlist[othername].diviname == div:
    #            divlist.append(othername)
    #    return divlist

    #def get_conf_list(self, teamname):
    #    conf = self.teamlist[teamname].confiname
    #    confilist = list()
    #    for othername in self.teamlist:
    #        if self.teamlist[othername].confiname == conf:
    #            confilist.append(othername)
    #    return confilist

    #def sort_by_win(self, teamnamelist):
    #    for i in reverse(range(0,len(teamnamelist))):
    #        for j in range(0,i):
    #            if (get_win(teamnamelist[j]) < (get_win(teamnamelist[j+1]):
    #                temp = teamnamelist[j]
    #                teamnamelist[j] = teamnamelist[j+1]
    #                teamnamelist[j+1] = temp
    #    return teamnamelist

    def get_lose(self, teamname, teamnamelist):
        win, lose, remain, total = self.get_win_lose_remain_total(teamname, teamnamelist)
        return lose

    def sort_by_lose(self, teamnamelist):
        totalteamlist = self.totalnamelist
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

    #def must_div_lead(self, teamname):
    #    tot = self.totalnamelist
    #    divlist = self.get_divi_list(teamname)
    #    sortedlist = self.sort_by_lose(divlist)
    #    t = sortedlist[0]
    #    if (t == teamname) and (self.get_lose(t, tot) != 
    #                            self.get_lose(sortedlist[1], tot)):
    #        return True
    #    else:
    #        return False

    def win_all(self, teamname):
        for i in range(1, self.lenth+1):
            if (((self.gamelist[i].home == teamname) or (self.gamelist[i].away == teamname)) and
                (self.gamelist[i].win == '')):
                self.gamelist[i].win = teamname
        return self




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


    def run(self):
        
        def hope(teamnow, sbn):

            def step1(i, times, teamnow, sbn):
                conflist = self.get_conf_list(teamnow)
                listscore, scores = sbn.sort_by_lose(conflist)
                if times<7:
                    topi = i
                    if listscore[topi] == teamnow:
                        topi += 1
                        i += 1
                    while scores[i] == scores[i+1]:
                        i += 1
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
                            sbn2 = sbn2.win_all(listscore[topi])
                            if step1(topi+1, times+1, teamnow, sbn2):
                                return True
                    return False
                else:
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
                        return True
                    else:
                        if scores[position] > scores[7]:
                            ###
                            if teamnow == 'Portland Trail Blazers':
                                print(position, scores[position], scores[7])
                            return False
                        else:
                            tielist = list()
                            for ii in range(0,len(listscore)):
                                if scores[ii] == scores[position]:
                                    tielist.append(listscore[ii])
                            return tie_d(teamnow, tielist, sbn)
                

            def tie_d(teamnow, tielist, sbn):
                return True
            
            sbn = sbn.win_all(teamnow)
            times = 0
            i = 0
            return step1(i, times, teamnow, sbn)

        def find_last(scorelist, scores, outlist):
            last = list()
            flag = True
            for i in reversed(range(len(scorelist))):
                if not flag:
                    if scores[i] == scores[i+1]:
                        last.append(scorelist[i])
                    else:
                        break
                if flag and i>7:
                    if scorelist[i] not in outlist:
                        last.append(scorelist[i])
                        flag = False
            return last
        
        now = self.date
        end = self.wholegame.gamelist[len(self.wholegame.gamelist)].date
        westlist = self.get_conf_list('Boston Celtics')
        eastlist = self.get_conf_list('Golden State Warriors')
        while now <= end:
            nowgame = copy.deepcopy(self.wholegame)
            sb = ScoreBoard(self.teamslist, nowgame, now)
            wscorelist, wscores = sb.sort_by_lose(westlist)
            escorelist, escores = sb.sort_by_lose(eastlist)
            wlast = find_last(wscorelist, wscores, self.outlist)
            elast = find_last(escorelist, escores, self.outlist)
            sbn = copy.deepcopy(sb)
            if wlast != []:
                for last in wlast:
                    ###
                    if last == 'Portland Trail Blazers':
                        print(now)
                        for i in range(len(scores)):
                            print(scorelist[i], scores[i])
                    if not hope(last, sbn):
                        self.outlist.append(last)
                        self.outlistdate[last] = now
            sbn = copy.deepcopy(sb)
            if elast != []:
                for last in elast:
                    if not hope(last, sbn):
                        self.outlist.append(last)
                        self.outlistdate[last] = now
            now = now + datetime.timedelta(days = 1)
        for teamname in self.totalnamelist:
            if teamname not in self.outlist:
                self.outlist.append(teamname)
                self.outlistdate[teamname] = 'Playoff'
        for ele in self.outlistdate:
            print(ele, self.outlistdate[ele])
    
    def get_outlist(self):
        return self.outlistdate
    
if __name__ == '__main__':
    team = TeamsList()
    game = GamesList()
    season = Season(team, game)
    season.run()
    season.get_outlist()




















