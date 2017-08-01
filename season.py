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
                    if ((othername == self.gamelist[i].home) or
                        (othername == self.gamelist[i].away) and 
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

    def get_divi_list(self, teamname):
        div = self.teamlist[teamname].diviname
        divlist = list()
        for othername in self.teamlist:
            if self.teamlist[othername].diviname == div:
                divlist.append(othername)
        return divlist

    def get_conf_list(self, teamname):
        conf = self.teamlist[teamname].confiname
        confilist = list()
        for othername in self.teamlist:
            if self.teamlist[othername].confiname == conf:
                confilist.append(othername)
        return confilist
    '''
    def sort_by_win(self, teamnamelist):
        for i in reverse(range(0,len(teamnamelist))):
            for j in range(0,i):
                if (get_win(teamnamelist[j]) < (get_win(teamnamelist[j+1]):
                    temp = teamnamelist[j]
                    teamnamelist[j] = teamnamelist[j+1]
                    teamnamelist[j+1] = temp
        return teamnamelist
    '''
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
        #for i in range(0, len(teamnamelist)):
        #    print(teamnamelist[i], self.get_win_lose_remain_total(teamnamelist[i], totalteamlist))
        return teamnamelist

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
        pass




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

        
    
        now = self.date
        end = self.wholegame.gamelist[len(self.wholegame.gamelist)].date
        westlist = self.get_conf_list('Boston Celtics')
        eastlist = self.get_conf_list('Golden State Warriors')

    def run(self):
        
        def hope(self, teamname):

            def step1(self, teamname):
                pass

            def step2(self, teamname):
                pass

            return True

        def find_last(scorelist, outlist):
            last = ''
            i = len(scorelist)
            while i>7:
                if scorelist[i] not in outlist:
                    last.appned(scorelist[i])


                i -= 1


            

            return last
        
        now = self.date
        end = self.wholegame.gamelist[len(self.wholegame.gamelist)].date
        westlist = self.get_conf_list('Boston Celtics')
        eastlist = self.get_conf_list('Golden State Warriors')
        while now <= end:
            nowgame = copy.deepcopy(self.wholegame)
            sb = ScoreBoard(self.teamslist, nowgame, now)
            wscorelist = sb.sort_by_lose(westlist)
            escorelist = sb.sort_by_lose(eastlist)
            wlast = find_last(wscorelist, self.outlist)
            elast = find_last(escorelist, self.outlist)
            for last in wlast:
                if not hope(last):
                    self.outlist.append(last)
                    self.outlistdate[last] = now
            for last in elast:
                if not hope(last):
                    self.outlist.append(last)
                    self.outlistdate[last] = now
            now = now + datetime.timedelta(days = 1)
        for teamname in self.totalnamelist:
            if teamname not in self.outlist:
                self.outlist.append(teamname)
                self.outlistdate[teamname] = 'playoff'
        print(self.outlistdate)
    
    def get_outlist(self):
        return self.outlistdate
    
if __name__ == '__main__':
    team = TeamsList()
    game = GamesList()
    season = Season(team, game)
    season.run()
    season.get_outlist()


























