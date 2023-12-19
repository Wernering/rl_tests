# Standard Library
import math

# External
import pygame
from algorithm import AIGame
from game_config import (
    ALPHA,
    BLACK,
    BLUE,
    BORDER,
    CARDINALITY_MOVESET,
    ENDING_POINT,
    EPSILON,
    GREEN,
    GRID_HEIGHT,
    GRID_WIDTH,
    GRIDWORLD_COLUMNS,
    GRIDWORLD_ROWS,
    GRIDWORLD_WIND,
    MOVESET_POSITION,
    ORANGE,
    POSITION_FIX,
    STARTING_POINT,
    TD_METHOD,
    WHITE,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)


class PlayerRect:
    def __init__(self, x_size, y_size, init_x, init_y, color, margin=10):
        self.x_size = x_size
        self.y_size = y_size

        self.color = color

        self.margin = margin

        self.Rect = pygame.Rect(
            self.adapt_position(init_x, self.x_size, self.margin),
            self.adapt_position(init_y, self.y_size, self.margin),
            self.x_size,
            self.y_size,
        )

    def update_position(self, x, y):
        self.Rect = pygame.Rect(
            self.adapt_position(x, self.x_size, self.margin),
            self.adapt_position(y, self.y_size, self.margin),
            self.x_size,
            self.y_size,
        )

    @staticmethod
    def adapt_position(val, size, margin):
        """Adapts the position of the player to the pygame Grid dimensions"""
        return val * size + margin


class VisualGame:
    def __init__(
        self,
        gw_rows,
        gw_cols,
        gw_wind,
        sp,
        ep,
        agent_alpha,
        agent_epsilon,
        agent_evt1,
        agent_card_mov,
    ):
        pygame.init()

        self.clock = pygame.time.Clock()

        # Columns and rows of the GridWorld
        self.gw_rows = gw_rows
        self.gw_cols = gw_cols

        # Screen Configs
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.screen.fill(WHITE)
        pygame.display.set_caption("Reinforcement Learning - GridWorld")

        # For status of the simulation
        self.font1 = pygame.font.SysFont("arial", 40)

        # For Arrows/wind
        self.font2 = pygame.font.SysFont("arialblack", 15)

        # For
        self.font3 = pygame.font.SysFont("arial", 15)

        # Size of blocks
        self.xblocksize, self.yblocksize = self.define_block_size()

        # Agent playing the simulation
        self.AI = AIGame(
            gw_rows,
            gw_cols,
            gw_wind,
            sp,
            ep,
            agent_alpha,
            agent_epsilon,
            agent_evt1,
            agent_card_mov,
        )

        # Rectangle that represents the player
        self.player = PlayerRect(self.xblocksize, self.yblocksize, self.AI.Agent.cx, self.AI.Agent.cy, GREEN)

        # Image for the wind arrows
        self.arrow_image = pygame.image.load("arrow.png").convert_alpha()
        self.arrow_image = pygame.transform.scale(
            self.arrow_image, (int(self.xblocksize / 2), int(self.yblocksize / 2))
        )
        self.arrow_image.set_alpha(127)

    def define_block_size(self):
        xblocksize = int(GRID_WIDTH / (self.gw_cols + 1))
        yblocksize = int(GRID_HEIGHT / (self.gw_rows + 1))
        return xblocksize, yblocksize

    def draw_rectangle(self, x, y, color, line_size):
        x = x * self.xblocksize + BORDER
        y = y * self.yblocksize + BORDER
        rect = pygame.Rect(x, y, self.xblocksize, self.yblocksize)
        pygame.draw.rect(self.screen, color, rect, line_size)
        return rect

    def draw_grid(self, wm=False, sa_matrix=False):
        """
        Draw the grid of the simulation.

        :param wm:
        :param sa_matrix:
        :return:
        """
        for x in range(self.gw_cols + 1):
            for y in range(self.gw_rows + 1):
                rect = self.draw_rectangle(x, y, BLACK, 1)

                # Draw state-action matrix
                if sa_matrix:
                    for key in self.AI.Agent.moveset.keys():
                        text = self.font3.render(f"{self.AI.Agent.sa_matrix[key, y, x]:.1f}", True, BLACK)
                        pos_text = text.get_rect()

                        new_pos = tuple(i * j for i, j in zip(getattr(rect, MOVESET_POSITION[key]), POSITION_FIX[key]))
                        setattr(pos_text, MOVESET_POSITION[key], new_pos)
                        self.screen.blit(text, pos_text)

                # Draw arrows to show direction.
                elif self.AI.GW.wind_matrix[0, y, x] != 0 or self.AI.GW.wind_matrix[1, y, x] != 0:
                    if not wm:
                        arrow = self.arrow_image.copy()
                        angle = (
                            -math.atan2(self.AI.GW.wind_matrix[1, y, x], self.AI.GW.wind_matrix[0, y, x])
                            * 180
                            / math.pi
                        )
                        arrow = pygame.transform.rotate(arrow, angle)
                        arrow_pos = arrow.get_rect(center=rect.center)
                        self.screen.blit(arrow, arrow_pos)

                    if wm:
                        # Draw text equals to the force of the wind (matrix transformation)
                        text = self.font2.render(
                            f"({self.AI.GW.wind_matrix[1, y, x]},{self.AI.GW.wind_matrix[0, y, x]})",
                            True,
                            BLACK,
                        )
                        pos_text = text.get_rect()
                        pos_text.center = rect.center
                        self.screen.blit(text, pos_text)

    def draw_best_route(self):
        if self.AI.Agent.optimal_route:
            list_points = [
                [
                    x * self.xblocksize + int(self.xblocksize / 2) + BORDER,
                    y * self.yblocksize + int(self.yblocksize / 2) + BORDER,
                ]
                for [y, x], _ in self.AI.Agent.optimal_route
            ]
            for coord in list_points:
                pygame.draw.circle(self.screen, BLUE, coord, 3)
            pygame.draw.lines(self.screen, BLUE, False, list_points, 3)

    def draw_goal(self, line_size):
        self.draw_rectangle(self.AI.ep[1], self.AI.ep[0], ORANGE, line_size)

    def draw_player(self):
        pygame.draw.rect(self.screen, self.player.color, self.player.Rect, 0)
        pygame.draw.rect(self.screen, BLACK, self.player.Rect, 1)

    def game_info(self):
        text1 = self.font1.render(f"Cycle: {self.AI.total_cycles}", True, BLACK)
        text2 = self.font1.render(f"Cycle_step: {self.AI.cycle_iterations}", True, BLACK)
        text3 = self.font1.render(f"Best_route_steps: {self.AI.return_optimal_route_steps()}", True, BLACK)
        text4 = self.font1.render(f"Total_steps: {self.AI.total_iterations}", True, BLACK)

        text1_rect = text1.get_rect()
        text2_rect = text2.get_rect()
        text3_rect = text3.get_rect()
        text4_rect = text4.get_rect()

        text_separation = int(
            (
                self.screen.get_width()
                - 2 * BORDER
                - (text1.get_width() + text2.get_width() + text3.get_width() + text4.get_width())
            )
            / 4
        )

        text1_rect.topleft = (BORDER, GRID_HEIGHT + 3 * BORDER)
        text2_rect.topleft = (text1_rect.topright[0] + text_separation, GRID_HEIGHT + 3 * BORDER)
        text3_rect.topleft = (text2_rect.topright[0] + text_separation, GRID_HEIGHT + 3 * BORDER)
        text4_rect.topleft = (text3_rect.topright[0] + text_separation, GRID_HEIGHT + 3 * BORDER)

        self.screen.blit(text1, text1_rect)
        self.screen.blit(text2, text2_rect)
        self.screen.blit(text3, text3_rect)
        self.screen.blit(text4, text4_rect)

    def loop(self):
        pause = True
        fps = [2, 4, 10, 20, 60, 120, 240]
        current_fps = len(fps) - 1
        show_wm = False
        sa_matrix = True
        best_route = False

        while True:
            # Speed - fps
            self.clock.tick(fps[current_fps])

            # White Screen to "refresh"
            self.screen.fill(WHITE)

            # Draw state
            self.draw_goal(0)
            self.draw_player()
            self.draw_grid(wm=show_wm, sa_matrix=sa_matrix)

            if best_route:
                self.draw_best_route()

            self.game_info()

            if not pause:
                # Let AI make a move
                if not self.AI.check_victory():
                    self.AI.turn(ui=True)
                else:
                    self.AI.Agent.set_position(*self.AI.sp)

                # Update position of player
                self.player.update_position(self.AI.Agent.cx, self.AI.Agent.cy)

            # For different events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        pause = not pause

                    if event.key == pygame.K_UP:
                        current_fps = min(current_fps + 1, len(fps) - 1)

                    if event.key == pygame.K_DOWN:
                        current_fps = max(current_fps - 1, 0)

                    if event.key == pygame.K_g:
                        if self.AI.Agent.epsilon == EPSILON:
                            self.AI.Agent.update_epsilon(0)
                        else:
                            self.AI.Agent.update_epsilon(EPSILON)

                    if event.key == pygame.K_w:
                        show_wm = not show_wm

                    if event.key == pygame.K_m:
                        sa_matrix = not sa_matrix

                    if event.key == pygame.K_s:
                        mods = pygame.key.get_mods()
                        if mods & pygame.KMOD_CTRL:
                            pause = True
                            self.AI.graph(show=False, save=True)

                    if event.key == pygame.K_t:
                        best_route = not best_route

            pygame.display.update()


if __name__ == "__main__":
    Game = VisualGame(
        gw_rows=GRIDWORLD_ROWS,
        gw_cols=GRIDWORLD_COLUMNS,
        gw_wind=GRIDWORLD_WIND,
        sp=STARTING_POINT,
        ep=ENDING_POINT,
        agent_alpha=ALPHA,
        agent_epsilon=EPSILON,
        agent_evt1=TD_METHOD,
        agent_card_mov=CARDINALITY_MOVESET,
    )

    Game.loop()
