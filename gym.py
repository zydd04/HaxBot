import random

class PlayerAction:
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    SHOOT = 4
class FieldObj:
    WALL = -1
    PLAYER = 2
    BALL = 0

class Field:
    def __init__(self, field_h = 26, field_w = 62):
        self.field_h = field_h
        self.field_w = field_w

    def reset(self, seed = None):
        self.player_pos = [20, 13]
        random.seed(seed)
        self.ball_pos = [random.randint(1, self.field_w-1), random.randint(1, self.field_h-1)]

    def perform_action(self, player_action:PlayerAction) -> bool:
        if player_action == PlayerAction.LEFT and self.player_pos[0] > 0:
            self.player_pos[0] -= 1
        elif player_action == player_action.RIGHT and self.player_pos[0] < self.field_w-1:
            self.player_pos[0] += 1
        elif player_action == player_action.UP and self.player_pos[1] < 26:
            self.player_pos[1] += 1
        elif player_action == player_action.DOWN and self.player_pos[1] > 0:
            self.player_pos[1] -= 1
        elif player_action == player_action.SHOOT:
            if self.player_pos[0] == self.ball_pos[0]:
                self.ball_pos[0] = max(self.ball_pos[0]+10, self.field_w)
            elif self.player_pos[1] == self.ball_pos[1]:
                self.ball_pos[1] = max(self.ball_pos[1]+10, self.field_h)

    def renderfield(self):
        for r in range(self.field_w):
            for c in range(self.field_h):
                if (r < 62 or r > 0) and (c == 0 or c == 26):
                    print(FieldObj.WALL, end = ' ')
                elif (c > 0 and (c < 10 or c > 16) and c < 26) and (r == 0 or r == 62):
                    print(FieldObj.WALL, end = ' ') 
                if ([r,c] == self.player_pos):
                    print(FieldObj.PLAYER, end = ' ')
                elif ([r,c] == self.ball_pos):
                    print(FieldObj.BALL, end = ' ')
                print()
            print()
if __name__ == "__main__":
    Field = Field()
    Field.renderfield()
    for i in range(20):
        randact = random.choice(list(PlayerAction))
        print(randact)
        Field.perform_action(randact)
        Field.renderfield()


                



