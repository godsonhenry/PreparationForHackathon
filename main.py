# file name is main
from readwrite import read_teams, read_games, write_outlist
from season import Season


class NBAPredictout():
    '''
    '''
    def main():
        teamslist= read_teams('') #files 
        gameslist = read_games('') #
        se = Season(teamslist,gameslist)
        se.run()
        outlist = se.get_outlist()
        write_outlist(outlist) #write the list out

    if __name__ == '__main__':
        main()
        



