import unittest
import season

class TestSeason(unittest.TestCase):
    def test_init(self):
        pass
    def test_ScoreBoard(self):
        team = Teams()
        game = Games()
        se_t = Season(team, game)
        sb_t = se_t.ScoreBoard()
        self.assertEqual(6, sb_t.get_win(''))
    
    if __name__ == '__main__':
        unittest.main()


