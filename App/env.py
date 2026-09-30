"""Two-snake environment: Snake 1 gathers food, Snake 2 hunts Snake 1."""
import random

import numpy as np
import pygame

from config import BLOCK_SIZE, WINDOW_H, WINDOW_W

# Direction encoding: 0 = Left, 1 = Right, 2 = Up, 3 = Down
LEFT, RIGHT, UP, DOWN = 0, 1, 2, 3
CLOCKWISE = [RIGHT, DOWN, LEFT, UP]

# Rewards
R_FOOD = 100.0            # S1 吃到食物
R_ENEMY_FOOD = -100.0     # S1 吃到食物時 S2 的懲罰
R_DIE_ALONE = -50.0       # 自己撞牆 / 撞自己
R_BLOCKED = -100.0        # S1 撞上 S2 的身體
R_BLOCK_SUCCESS = 200.0   # S2 阻擋成功
R_HEAD_ON_LOSS = -100.0   # 頭對頭：S1
R_HEAD_ON_WIN = 100.0     # 頭對頭：S2
R_BOTH_DIE = -10.0        # 非頭對頭的同時死亡


class MultiSnakeGameAI:
    def __init__(self, w=WINDOW_W, h=WINDOW_H, render=True, fps=100, max_steps=None):
        self.w = w
        self.h = h
        self.render = render
        self.fps = fps
        self.max_steps = max_steps
        self.quit_requested = False  # 使用者關閉視窗時設為 True，由主迴圈負責收尾與存檔
        if self.render:
            # 只初始化需要的模組；pygame.init() 會連音效一起初始化，在沒有音效卡的環境 (如 Docker) 會噴 ALSA 警告
            pygame.display.init()
            pygame.font.init()
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake DQN: Gatherer (S1) vs. Hunter (S2)')
            self.font = pygame.font.SysFont('arial', 20)
            self.clock = pygame.time.Clock()
        self.reset_all()

    def close(self):
        if self.render:
            pygame.quit()

    def reset_all(self):
        """重置整局，包含分數。"""
        self.score1 = 0
        self.score2 = 0
        self.respawn(1)
        self.respawn(2)
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        return self.get_state(1), self.get_state(2)

    def respawn(self, snake_id):
        """讓特定的蛇復活 (重置位置與長度)，保留分數。"""
        if snake_id == 1:
            self.direction1 = RIGHT
            self.head1 = [self.w // 4, self.h // 2]
            self.snake1 = [list(self.head1),
                           [self.head1[0] - BLOCK_SIZE, self.head1[1]],
                           [self.head1[0] - 2 * BLOCK_SIZE, self.head1[1]]]
        else:
            self.direction2 = LEFT
            self.head2 = [3 * self.w // 4, self.h // 2]
            self.snake2 = [list(self.head2),
                           [self.head2[0] + BLOCK_SIZE, self.head2[1]],
                           [self.head2[0] + 2 * BLOCK_SIZE, self.head2[1]]]

    def _place_food(self):
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            candidate = [x, y]
            if candidate not in self.snake1 and candidate not in self.snake2:
                self.food = candidate
                break

    def play_step(self, action1_idx, action2_idx):
        """回傳: state1, state2, reward1, reward2, dead1, dead2"""
        self.frame_iteration += 1

        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit_requested = True

        # 1) 計算兩蛇的新 head
        next_head1 = self._calc_next_head(action1_idx, 1)
        next_head2 = self._calc_next_head(action2_idx, 2)

        # S1 的下一步是否撞上 S2 的身體 (用於獎勵 S2 的阻擋)
        snake1_hit_snake2_body = next_head1 in self.snake2[:-1]

        # 2) 碰撞判定
        c1 = self._will_collision(next_head1, snake_id=1, next_head_other=next_head2)
        c2 = self._will_collision(next_head2, snake_id=2, next_head_other=next_head1)

        # 頭對頭：兩顆頭移到同一格，或兩顆頭互換位置 (穿過彼此)
        heads_swapped = next_head1 == self.head2 and next_head2 == self.head1
        head_on_crash = next_head1 == next_head2 or heads_swapped

        reward1 = 0.0
        reward2 = 0.0
        dead1 = c1
        dead2 = c2

        # 3) 死亡與復活
        if c1 and c2:
            if head_on_crash:
                # S2 頭對頭獲勝 (風險比阻擋高，獎勵給少一點)
                reward1 = R_HEAD_ON_LOSS
                reward2 = R_HEAD_ON_WIN
                self.score2 += 1
            else:
                reward1 = R_BOTH_DIE
                reward2 = R_BOTH_DIE
            self.respawn(1)
            self.respawn(2)
        elif c1:
            if snake1_hit_snake2_body:
                # 阻擋成功：S1 撞上 S2 的身體
                reward1 = R_BLOCKED
                reward2 = R_BLOCK_SUCCESS
                self.score2 += 1
            else:
                reward1 = R_DIE_ALONE
            self.respawn(1)
        elif c2:
            reward2 = R_DIE_ALONE
            self.respawn(2)

        # 4) 沒死的蛇往前移動
        if not dead1:
            self.snake1.insert(0, list(next_head1))
            self.head1 = list(next_head1)
        if not dead2:
            self.snake2.insert(0, list(next_head2))
            self.head2 = list(next_head2)

        # 5) S1：吃到食物就變長 (不 pop 尾巴)
        if not dead1 and self.head1 == self.food:
            reward1 += R_FOOD
            # 用加的而不是覆蓋：若 S2 同一步自己撞死，它的死亡懲罰不會被蓋掉
            reward2 += R_ENEMY_FOOD
            self.score1 += 1
            self._place_food()
        elif not dead1:
            self.snake1.pop()

        # 6) S2：只有阻擋成功才變長，食物對 S2 沒有作用
        if not dead2 and not snake1_hit_snake2_body:
            self.snake2.pop()

        if self.render:
            self._update_ui()
            self.clock.tick(self.fps)

        return self.get_state(1), self.get_state(2), reward1, reward2, dead1, dead2

    def _calc_next_head(self, action_idx, snake_id):
        """action: 0 = 直走, 1 = 右轉, 2 = 左轉。會同時更新該蛇的方向。"""
        if snake_id == 1:
            curr_dir, head = self.direction1, self.head1
        else:
            curr_dir, head = self.direction2, self.head2

        idx = CLOCKWISE.index(curr_dir)
        if action_idx == 1:
            idx = (idx + 1) % 4
        elif action_idx == 2:
            idx = (idx - 1) % 4
        new_dir = CLOCKWISE[idx]

        if snake_id == 1:
            self.direction1 = new_dir
        else:
            self.direction2 = new_dir

        x, y = head
        if new_dir == RIGHT:
            x += BLOCK_SIZE
        elif new_dir == LEFT:
            x -= BLOCK_SIZE
        elif new_dir == DOWN:
            y += BLOCK_SIZE
        elif new_dir == UP:
            y -= BLOCK_SIZE
        return [x, y]

    def _will_collision(self, next_head, snake_id, next_head_other=None):
        # 1. 撞牆
        if (next_head[0] > self.w - BLOCK_SIZE or next_head[0] < 0
                or next_head[1] > self.h - BLOCK_SIZE or next_head[1] < 0):
            return True

        # 2. 撞到頭 (Head-on)
        if next_head_other is not None and next_head == next_head_other:
            return True

        # 3. 撞到任一條蛇的身體 (尾巴這一步會移走，所以不算)
        return next_head in self.snake1[:-1] or next_head in self.snake2[:-1]

    def get_state(self, snake_id):
        if snake_id == 1:
            head, direction = self.snake1[0], self.direction1
            target = self.food            # S1 的目標：食物
        else:
            head, direction = self.snake2[0], self.direction2
            target = self.snake1[0]       # S2 的目標：S1 的頭

        point_l = [head[0] - BLOCK_SIZE, head[1]]
        point_r = [head[0] + BLOCK_SIZE, head[1]]
        point_u = [head[0], head[1] - BLOCK_SIZE]
        point_d = [head[0], head[1] + BLOCK_SIZE]

        dir_l = direction == LEFT
        dir_r = direction == RIGHT
        dir_u = direction == UP
        dir_d = direction == DOWN

        def danger(p):
            return self._will_collision(p, snake_id)

        state = [
            # 1-3. 危險：前方 / 右方 / 左方 (相對於自己的方向)
            (dir_r and danger(point_r)) or (dir_l and danger(point_l)) or
            (dir_u and danger(point_u)) or (dir_d and danger(point_d)),

            (dir_u and danger(point_r)) or (dir_d and danger(point_l)) or
            (dir_l and danger(point_u)) or (dir_r and danger(point_d)),

            (dir_d and danger(point_r)) or (dir_u and danger(point_l)) or
            (dir_r and danger(point_u)) or (dir_l and danger(point_d)),

            # 4-7. 自己目前的方向
            dir_l, dir_r, dir_u, dir_d,

            # 8-11. 目標的相對方向 (S1: 食物, S2: S1 的頭)
            target[0] < head[0], target[0] > head[0],
            target[1] < head[1], target[1] > head[1],

            # 12-15. S1 目前的方向
            self.direction1 == LEFT, self.direction1 == RIGHT,
            self.direction1 == UP, self.direction1 == DOWN,
        ]
        return np.array(state, dtype=np.int32)

    def _update_ui(self):
        self.display.fill((0, 0, 0))

        for pt in self.snake1:  # Snake 1 (Blue)
            pygame.draw.rect(self.display, (0, 0, 255), pygame.Rect(pt[0], pt[1], BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, (0, 100, 255), pygame.Rect(pt[0] + 4, pt[1] + 4, 12, 12))

        for pt in self.snake2:  # Snake 2 (Green)
            pygame.draw.rect(self.display, (0, 255, 0), pygame.Rect(pt[0], pt[1], BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, (0, 200, 0), pygame.Rect(pt[0] + 4, pt[1] + 4, 12, 12))

        pygame.draw.rect(self.display, (200, 0, 0),  # Food (Red)
                         pygame.Rect(self.food[0], self.food[1], BLOCK_SIZE, BLOCK_SIZE))

        hud = f"S1(Food): {self.score1} | S2(Blocks): {self.score2}"
        if self.max_steps:
            hud = f"Steps left: {max(0, self.max_steps - self.frame_iteration)} | " + hud
        self.display.blit(self.font.render(hud, True, (255, 255, 255)), [0, 0])

        pygame.display.flip()
