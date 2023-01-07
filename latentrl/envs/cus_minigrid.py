import copy
import math

import numpy as np
import time
import sys
import random
from enum import IntEnum
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Lava, Ball, Box, Door, Wall, WorldObj
from minigrid.core.mission import MissionSpace
from minigrid.core.constants import (
    COLOR_TO_IDX,
    COLORS,
    IDX_TO_COLOR,
    IDX_TO_OBJECT,
    OBJECT_TO_IDX,
)
from minigrid.utils.rendering import (
    fill_coords,
    point_in_circle,
    point_in_line,
    point_in_rect,
)


class CollectFlags(MiniGridEnv):
    # class Actions(IntEnum):
    #     # Turn left, turn right, move forward
    #     left = 0
    #     right = 1
    #     forward = 2
    #     # Pick up an object
    #     # pickup = 3
    #     # Drop an object
    #     # drop = 4
    #     # Toggle/activate an object
    #     toggle = 5
    #     # Done completing task
    #     # done = 6

    def __init__(
        self,
        maze_name: str = "basic",
        max_steps: int = 8000,
        scaled: None | int = None,
        stochasticity: dict = None,
        **kwargs,
    ):
        self.maze_name = maze_name
        self.scaled = scaled
        self.stochasticity = stochasticity
        (self.room_layout, self.agent_pos_start, self.flags_pos, self.goal_pos,) = get_room_layout(
            maze_name=maze_name
        )  # return a nparray
        self.width = self.room_layout.shape[1] + 2
        self.height = self.room_layout.shape[0] + 2
        template = np.full((self.height, self.width), "w")
        template[1:-1, 1:-1] = self.room_layout
        self.room_layout = template
        self.agent_pos_start = (self.agent_pos_start[0] + 1, self.agent_pos_start[1] + 1)
        self.flags_pos = np.array(self.flags_pos) + 1
        self.goal_pos = np.array(self.goal_pos) + 1

        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            width=self.width,
            height=self.height,
            see_through_walls=False,  # Set this to True for maximum speed
            max_steps=max_steps,
            **kwargs,
        )

        opens_original = np.argwhere(self.room_layout == "z").tolist()
        self.opens_original = {str(i) for i in opens_original}
        if self.scaled:
            newLayout = []
            for l in self.room_layout:
                nextLine = []
                for x in l:
                    nextLine.extend([x for _ in range(self.scaled)])
                newLayout.extend([nextLine for _ in range(self.scaled)])
                # newLayout.append(nextLine)
                # newLayout.append(nextLine)
                # newLayout.append(nextLine)
            self.room_layout = np.array(newLayout)

            self.agent_pos_start = tuple([i * self.scaled for i in self.agent_pos_start])
            self.goal_pos = tuple([i * self.scaled for i in self.goal_pos])
            flags_pos_scaled = []
            for pos in self.flags_pos:
                flags_pos_scaled.append((pos[0] * self.scaled, pos[1] * self.scaled))
            self.flags_pos = flags_pos_scaled

        self.walls_pos = np.argwhere(self.room_layout == "w").tolist()
        self.walls_pos_original = copy.deepcopy(self.walls_pos)
        # self.walls can be updated, since stochastic opens can become walls time to time
        self.traps_pos = np.argwhere(self.room_layout == "t").tolist()
        self.opens_pos = np.argwhere(self.room_layout == "z").tolist()

        self.print_maze_info()

    def print_maze_info(self):
        print("env.size:", (len(self.room_layout), len(self.room_layout[0])))
        print("env.name:", self.maze_name)
        print("env scaled:", self.scaled)
        print("env.flags_pos:", self.flags_pos)
        print("env.goal_pos:", self.goal_pos)
        print("env.agent_pos_start:", self.agent_pos_start)

    @staticmethod
    def _gen_mission():
        return "pick the keys and get to the goal."

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        # self.grid.wall_rect(0, 0, width, height)

        self.agent_pos = (self.agent_pos_start[1], self.agent_pos_start[0])
        self.agent_dir = 0

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), self.goal_pos[1], self.goal_pos[0])

        # Place Wall in the Grid
        for wall_pos in self.walls_pos:
            self.put_obj(Wall(), wall_pos[1], wall_pos[0])

        # Place keys in the Grid
        for key_pos in self.flags_pos:
            self.put_obj(Key(), key_pos[1], key_pos[0])

    def choose_stochastic_opens(self):
        if self.stochasticity:
            opens_2be_stochastic1 = set(
                random.sample(
                    self.opens_original,
                    math.floor(len(self.opens_original) * self.stochasticity["p2be_stochastic"]),
                )
            )
            opens_2be_stochastic2 = set()
            if self.scaled:
                for term in opens_2be_stochastic1:
                    for t1 in range(self.scaled):
                        for t2 in range(self.scaled):
                            x = (eval(term)[0]) * self.scaled + t1
                            y = (eval(term)[1]) * self.scaled + t2
                            opens_2be_stochastic2.add(f"[{x}, {y}]")
            else:
                opens_2be_stochastic2 = opens_2be_stochastic1
            self.opens_2be_stochastic2 = opens_2be_stochastic2

    def update_walls(self, move_count):
        if self.stochasticity and move_count % self.stochasticity["interval"] == 0:
            self.walls_pos = self.walls_pos_original
            if (
                np.random.randint(100) < self.stochasticity["p2be_closed"] * 100
            ):  # probability for chosen opens to become untraversible
                self.walls_pos = self.walls_pos_original.union(self.opens_2be_stochastic2)

    def reset(self, seed=None, options=None):

        # Reinitialize episode-specific variables
        self.agent_pos = (-1, -1)
        self.agent_dir = -1
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        assert (
            self.agent_pos >= (0, 0)
            if isinstance(self.agent_pos, tuple)
            else all(self.agent_pos >= 0) and self.agent_dir >= 0
        )

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        self.step_count = 0
        self.flags_carrying = 0
        if self.render_mode == "human":
            self.render()

        # Return first observation
        obs = self.gen_obs()

        return obs, {}

    def get_valid_coords_and_states(self):
        valid_coords = []
        valid_states = []
        template = self.room_layout
        templateX = len(template[0])  # num of columns
        templateY = len(template)  # num of rows
        for i in range(templateY):
            for j in range(templateX):
                if template[i, j] != "w":
                    current_coord = (i, j)
                    valid_coords.append(current_coord)
                else:
                    continue
                for k in range(2):
                    for l in range(2):
                        for m in range(2):
                            current_state = [i, j, k, l, m]
                            if current_coord in self.flags_pos:
                                index = self.flags_pos.index(current_coord)
                                current_state[2 + index] = 1
                            valid_states.append(tuple(current_state))

        return set(valid_coords), set(valid_states)
        # return list(set(valid_coords)), list(set(valid_states))

    def step(self, action):
        self.step_count += 1

        reward = -1e-6
        # reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = 0.1 * self.flags_carrying
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True

        # Pick up an object
        # elif action == self.actions.pickup:
        #     if fwd_cell and fwd_cell.can_pickup():
        #         if self.carrying is None:
        #             self.carrying = fwd_cell
        #             self.carrying.cur_pos = np.array([-1, -1])
        #             self.grid.set(fwd_pos[0], fwd_pos[1], None)

        # Drop an object
        # elif action == self.actions.drop:
        #     if not fwd_cell and self.carrying:
        #         self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
        #         self.carrying.cur_pos = fwd_pos
        #         self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                toggled = fwd_cell.toggle(self, fwd_pos)
                if toggled:
                    reward = 1
                    self.flags_carrying += 1

        # Done action (not used by default)
        # elif action == self.actions.done:
        #     pass
        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}


class Key(WorldObj):
    def __init__(self, color="red"):
        super().__init__("key", color)

    def toggle(self, env, pos):
        env.grid.set(pos[0], pos[1], None)
        return True

    # def render(self, img):
    #     """
    #     render as a flag
    #     """
    #     c = COLORS[self.color]

    #     # Outline
    #     fill_coords(img, point_in_rect(0.2, 0.8, 0.2, 0.5), c)
    #     # fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0, 0, 0))

    #     # # Horizontal slit
    #     # fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)

    #     # Stick for flag
    #     fill_coords(img, point_in_rect(0.2, 0.3, 0.2, 0.85), c)

    def render(self, img):
        """
        render as a key
        """
        c = COLORS[self.color]

        # Vertical quad
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)

        # Teeth
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)

        # Ring
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0, 0, 0))


# used for get_room_layout
A = 0
B = 1
C = 2
D = 3
E = 4
F = 5
G = 6
H = 7
I = 8
J = 9
K = 10
L = 11
M = 12
N = 13
O = 14
P = 15
Q = 16
R = 17
S = 18
T = 19
U = 20
V = 21
X = 22
Y = 23
Z = "z"
W = "w"


def get_room_layout(maze_name):
    if maze_name == "basic":
        ## "True" layout determined by doorways.
        room_layout = [
            [C, C, C, C, C, C, W, D, D, D, D, D, D, D, D, W, F, F, F, F, F],
            [C, C, C, C, C, C, W, D, D, D, D, D, D, D, D, W, F, F, F, F, F],
            [C, C, C, C, C, C, W, D, D, D, D, D, D, D, D, W, F, F, F, F, F],
            [C, C, C, C, C, C, W, D, D, D, D, D, D, D, D, W, F, F, F, F, F],
            [W, W, W, W, C, W, W, W, W, W, W, D, D, D, D, W, F, F, F, F, F],
            [B, B, B, B, B, B, B, E, E, E, W, D, D, D, D, W, F, F, F, F, F],
            [B, B, B, B, B, B, W, E, E, E, E, D, D, D, D, W, F, F, F, F, F],
            [B, B, B, B, B, B, W, E, E, E, W, D, D, D, D, W, F, F, F, F, F],
            [W, A, W, W, W, W, W, E, E, E, W, D, D, D, D, W, F, F, F, F, F],
            [A, A, A, A, A, A, W, E, E, E, W, D, D, D, D, W, F, F, F, F, F],
            [A, A, A, A, A, A, W, E, E, E, W, D, D, D, D, W, F, F, F, F, F],
            [A, A, A, A, A, A, W, W, E, W, W, W, W, W, W, W, F, F, F, F, F],
            [A, A, A, A, A, A, W, G, G, G, G, G, G, G, G, W, F, F, F, F, F],
            [A, A, A, A, A, A, W, G, G, G, G, G, G, G, G, W, F, F, F, F, F],
            [A, A, A, A, A, A, W, G, G, G, G, G, G, G, G, F, F, F, F, F, F],
            [A, A, A, A, A, A, W, G, G, G, G, G, G, G, G, W, F, F, F, F, F],
        ]
        room_layout = np.array(room_layout)
        # state = (6, 4, 0, 0, 0)          #v1
        # flags = [(0, 5), (15, 0), (15, 20)]
        # goal = (1

        agent_pos = (6, 1)  # v2
        flags_pos = [(0, 5), (0, 7), (4, 20)]
        goal_pos = (14, 1)

        # state = (6, 1, 0, 0, 0)        # v3
        # flags = [(0, 5), (1, 14), (1, 16)]
        # goal = (14, 1)

        # state = (6, 1, 0, 0, 0)        # v4
        # flags = [(0, 5), (15, 7), (1, 16)]
        # goal = (15, 18)

        # state = (15, 0, 0, 0, 0)    # v5
        # flags = [(10, 14), (0, 7), (0, 16)]
        # goal = (0, 5)
        # traps = [(15, 7), (15, 8)]

    elif maze_name == "basic2":
        ## "True" layout determined by doorways.
        room_layout = [
            [C, C, C, C, C, C, W, D, D, D, D, D, D, D, D, W, F, F, F, F, F],
            [C, C, C, C, C, C, W, D, D, D, D, D, D, D, D, W, F, F, F, F, F],
            [C, C, C, C, C, C, W, D, D, D, D, D, D, D, D, W, F, F, F, F, F],
            [C, C, C, C, C, C, W, D, D, D, D, D, D, D, D, W, F, F, F, F, F],
            [W, W, W, W, Z, W, W, W, W, W, W, D, D, D, D, W, F, F, F, F, F],
            [B, B, B, B, B, B, Z, E, E, E, W, D, D, D, D, W, F, F, F, F, F],
            [B, B, B, B, B, B, W, E, E, E, Z, D, D, D, D, W, F, F, F, F, F],
            [B, B, B, B, B, B, W, E, E, E, W, D, D, D, D, W, F, F, F, F, F],
            [W, Z, W, W, W, W, W, E, E, E, W, D, D, D, D, W, F, F, F, F, F],
            [A, A, A, A, A, A, W, E, E, E, W, D, D, D, D, W, F, F, F, F, F],
            [A, A, A, A, A, A, W, E, E, E, W, D, D, D, D, W, F, F, F, F, F],
            [A, A, A, A, A, A, W, W, Z, W, W, W, W, W, W, W, F, F, F, F, F],
            [A, A, A, A, A, A, W, G, G, G, G, G, G, G, G, W, F, F, F, F, F],
            [A, A, A, A, A, A, W, G, G, G, G, G, G, G, G, W, F, F, F, F, F],
            [A, A, A, A, A, A, W, G, G, G, G, G, G, G, G, Z, F, F, F, F, F],
            [A, A, A, A, A, A, W, G, G, G, G, G, G, G, G, W, F, F, F, F, F],
        ]
        room_layout = np.array(room_layout)
        # self.state = (6, 4, 0, 0, 0)          #v1
        # self.flags = [(0, 5), (15, 0), (15, 20)]
        # self.goal = (14, 1)

        agent_pos = (6, 1)  # v2
        flags_pos = [(0, 5), (0, 7), (4, 20)]
        goal_pos = (14, 1)

        # self.state = (6, 1, 0, 0, 0)        # v3
        # self.flags = [(0, 5), (1, 14), (1, 16)]
        # self.goal = (14, 1)

        # self.state = (6, 1, 0, 0, 0)        # v4
        # self.flags = [(0, 5), (15, 7), (1, 16)]
        # self.goal = (15, 18)

        # self.state = (15, 0, 0, 0, 0)    # v5
        # self.flags = [(10, 14), (0, 7), (0, 16)]
        # self.goal = (0, 5)
        # self.traps = [(15, 7), (15, 8)]
    elif maze_name == "simple":
        room_layout = [
            [D, D, D, F, F, F, F, F, F, F, W, H, H, H, H, H],  ## simple2
            [D, D, D, W, F, F, F, F, F, F, H, H, H, H, H, H],
            [D, D, D, W, F, F, F, F, F, F, W, H, H, H, H, H],
            [D, W, W, W, W, W, W, W, W, W, W, H, H, H, H, H],
            [G, G, G, G, G, G, G, W, E, E, W, H, H, H, H, H],
            [G, G, G, G, G, G, G, G, E, E, W, H, H, H, H, H],
            [G, G, G, G, G, G, G, G, E, E, W, W, W, W, B, B],
            [W, W, W, G, G, G, G, W, E, E, E, E, E, W, B, B],
            [A, A, W, W, W, W, W, W, E, E, E, E, E, W, B, B],
            [A, A, C, C, C, C, C, C, E, E, E, E, E, W, B, B],
            [A, A, W, C, C, C, C, C, E, E, E, E, E, W, B, B],
        ]
        room_layout = np.array(room_layout)
        agent_pos = (2, 2)  # v2
        flags_pos = [(0, 1), (0, 2), (0, 3)]
        goal_pos = (1, 1)

    elif maze_name == "strips":
        room_layout = [
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G],  ## Strips
            [A, A, W, B, B, B, C, C, W, D, D, D, W, E, E, W, F, F, W, G],
            [A, A, W, B, B, B, C, C, W, D, D, D, W, E, E, W, F, F, W, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, F, G],
            [A, A, W, B, B, W, C, C, C, D, D, D, W, E, E, W, F, F, W, G],
            [A, A, W, B, B, W, C, C, C, D, D, D, W, E, E, W, F, F, W, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G],
            [A, A, A, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, E, F, F, W, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, E, F, F, W, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, D, E, E, W, F, F, W, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G],
        ]
        room_layout = np.array(room_layout)
        agent_pos = (0, 0)
        flags_pos = [(15, 11), (19, 0), (4, 19)]
        goal_pos = (18, 1)
    elif maze_name == "strips2":
        room_layout = [
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],  ## Strips
            [A, A, W, B, B, B, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
            [A, A, W, B, B, B, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, F, G, G],
            [A, A, W, B, B, W, C, C, C, D, D, D, W, E, E, W, F, F, W, G, G],
            [A, A, W, B, B, W, C, C, C, D, D, D, W, E, E, W, F, F, W, G, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
            [A, A, A, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, E, F, F, W, G, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, E, F, F, W, G, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, D, E, E, W, F, F, W, G, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
        ]
        room_layout = np.array(room_layout)
        agent_pos = (0, 0)
        flags_pos = [(18, 3), (15, 11), (19, 19)]
        goal_pos = (18, 1)
    elif maze_name == "strips3":
        room_layout = [
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],  ## Strips
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
            [A, A, W, B, B, Z, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, Z, G, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
            [A, A, W, B, B, W, C, C, Z, D, D, D, W, E, E, W, F, F, W, G, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
            [A, A, Z, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, Z, F, F, W, G, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, Z, E, E, W, F, F, W, G, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
            [A, A, W, B, B, W, C, C, W, D, D, D, W, E, E, W, F, F, W, G, G],
        ]
        room_layout = np.array(room_layout)
        agent_pos = (0, 0)
        flags_pos = [(18, 3), (15, 11), (19, 19)]
        goal_pos = (18, 1)
    elif maze_name == "spiral":
        room_layout = [
            [C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, B, B],  ## Spiral
            [C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, B, B],
            [D, D, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, B, B],
            [D, D, W, G, G, G, G, G, G, G, G, G, G, G, G, F, F, W, B, B],
            [D, D, W, G, G, G, G, G, G, G, G, G, G, G, G, F, F, W, B, B],
            [D, D, W, H, H, W, W, W, W, W, W, W, W, W, W, F, F, W, B, B],
            [D, D, W, H, H, W, K, K, K, K, K, K, J, J, W, F, F, W, B, B],
            [D, D, W, H, H, W, K, K, K, K, K, K, J, J, W, F, F, W, B, B],
            [D, D, W, H, H, W, L, L, W, W, W, W, J, J, W, F, F, W, B, B],
            [D, D, W, H, H, W, L, L, M, M, M, W, J, J, W, F, F, W, B, B],
            [D, D, W, H, H, W, L, L, M, M, M, W, J, J, W, F, F, W, B, B],
            [D, D, W, H, H, W, W, W, W, W, W, W, J, J, W, F, F, W, B, B],
            [D, D, W, H, H, I, I, I, I, I, I, I, I, I, W, F, F, W, B, B],
            [D, D, W, H, H, I, I, I, I, I, I, I, I, I, W, F, F, W, B, B],
            [D, D, W, W, W, W, W, W, W, W, W, W, W, W, W, F, F, W, B, B],
            [D, D, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W, B, B],
            [D, D, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W, B, B],
            [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, B, B],
            [A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
            [A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
        ]
        room_layout = np.array(room_layout)
        # self.state = (19, 0, 0, 0, 0)       #v1
        # self.flags = [(0, 19), (15, 6), (6, 6)]
        # self.goal = (13, 13)

        agent_pos = (19, 0)  # v2
        flags_pos = [(0, 19), (15, 6), (10, 10)]
        goal_pos = (0, 0)

        # self.state = (19, 0, 0, 0, 0)         # v3
        # self.flags = [(3, 3), (16, 16), (10, 10)]
        # self.goal = (16, 18)
    elif maze_name == "open_space":
        room_layout = [
            [C, C, C, C, C, C, C, C, C, C, C, C, C, E, E, E, E, W, W, W],  ## Open Space
            [W, W, W, W, W, C, C, C, C, C, C, C, C, E, E, W, E, E, W, W],
            [W, W, W, W, W, C, C, C, C, W, W, W, D, E, E, E, W, E, E, W],
            [W, W, W, W, W, C, C, D, D, D, D, D, D, E, E, E, W, W, E, E],
            [B, B, B, B, B, B, D, D, D, D, D, D, D, E, E, E, E, W, W, E],
            [B, B, B, B, B, B, D, D, D, D, D, D, D, E, E, E, E, E, E, E],
            [B, B, B, B, B, B, D, D, D, W, D, D, D, E, E, E, E, E, E, E],
            [B, B, B, B, B, B, D, D, W, W, W, D, D, E, E, E, E, E, E, E],
            [B, B, B, B, B, B, D, W, W, W, W, W, F, F, F, F, F, F, W, W],
            [W, W, W, B, B, B, H, H, W, W, W, F, F, F, F, F, F, W, W, W],
            [W, W, W, B, B, B, H, H, H, W, H, F, F, F, F, F, F, F, W, W],
            [A, A, A, B, B, B, H, H, H, H, H, H, W, W, G, G, G, G, G, G],
            [A, A, A, B, B, B, H, H, H, H, H, H, W, W, G, G, G, G, G, G],
            [A, A, A, B, B, B, H, H, H, H, H, G, G, G, G, G, G, G, G, G],
            [A, A, A, B, B, B, H, H, H, H, H, G, G, G, G, G, G, G, G, G],
            [A, A, A, W, W, W, W, H, H, H, W, G, G, G, G, G, G, G, G, G],
            [A, A, A, W, W, W, W, H, H, H, W, G, G, G, G, W, W, G, G, G],
            [A, A, A, W, W, W, W, H, H, H, W, G, G, G, G, W, W, G, G, G],
            [A, A, A, W, W, W, W, H, H, H, W, G, G, G, G, W, W, G, G, G],
            [A, A, A, A, A, A, A, H, H, H, G, G, G, G, G, W, W, G, G, G],
        ]
        room_layout = np.array(room_layout)
        agent_pos = (19, 0)
        # flags_pos = [(0, 0), (2, 17), (13, 14)]
        flags_pos = [(1, 8), (5, 4), (13, 14)]
        # agent_pos = (0, 0)
        # flags_pos = [(0, 1), (0, 2), (13, 14)]
        # goal_pos = (19, 3)  # v1
        goal_pos = (19, 19)  # v2
    elif maze_name == "high_connectivity":
        room_layout = [
            [A, A, A, W, H, H, H, I, I, I, I, I, I, I, W, J, J, J, J, J],  ##High Connectivity
            [A, A, A, W, H, H, H, W, I, I, I, I, I, I, J, J, J, J, J, J],
            [A, A, A, H, H, H, H, W, I, I, I, I, I, I, W, J, J, J, J, J],
            [A, A, A, W, W, W, H, W, W, W, I, W, Z, W, W, J, J, J, J, J],
            [A, A, A, W, G, G, G, G, G, W, K, W, P, P, W, J, J, J, J, J],
            [A, A, A, W, G, G, G, G, G, K, K, W, P, P, W, W, W, J, J, J],
            [A, A, A, G, G, G, G, G, G, W, K, W, P, P, P, P, W, Z, W, W],
            [A, A, A, W, W, W, W, G, G, W, K, W, P, P, P, P, P, P, P, P],
            [A, A, A, A, A, A, W, F, W, W, W, W, P, P, P, P, P, P, P, P],
            [A, A, A, A, A, A, F, F, F, F, F, W, P, P, P, P, P, P, P, P],
            [A, A, A, A, A, A, W, F, F, F, F, P, P, P, P, P, P, P, P, P],
            [W, W, W, Z, W, W, W, Z, W, W, W, W, P, W, W, W, W, P, W, W],
            [B, B, B, B, W, E, E, E, E, E, W, O, O, O, O, O, W, N, N, N],
            [B, B, B, B, W, E, E, E, E, E, W, O, O, O, O, O, O, N, N, N],
            [B, B, B, B, Z, E, E, E, E, E, W, O, O, O, O, O, W, N, N, N],
            [B, B, B, B, E, W, E, E, E, W, W, Z, W, W, O, O, W, N, N, N],
            [W, C, W, W, E, W, W, W, E, W, L, L, L, W, O, W, W, W, N, W],
            [C, C, C, W, W, W, D, D, D, W, L, L, L, W, W, M, M, M, M, M],
            [C, C, C, C, D, D, D, D, D, W, L, L, L, L, M, M, M, M, M, M],
            [C, C, C, W, W, W, D, D, D, L, L, L, L, L, W, M, M, M, M, M],
        ]
        room_layout = np.array(room_layout)
        agent_pos = (19, 0)
        flags_pos = [(0, 1), (2, 18), (5, 6)]
        goal_pos = (15, 0)
    elif maze_name == "low_connectivity":
        room_layout = [
            [A, A, A, W, H, H, H, I, I, I, I, I, I, I, W, J, J, J, J, J],  ##Low Connectivity
            [A, A, A, W, H, H, H, W, I, I, I, I, I, I, J, J, J, J, J, J],
            [A, A, A, W, H, H, H, W, I, I, I, I, I, I, W, J, J, J, J, J],
            [A, A, A, W, W, W, H, W, W, W, I, W, W, W, W, J, J, J, J, J],
            [A, A, A, W, G, G, G, G, G, W, K, W, P, P, W, J, J, J, J, J],
            [A, A, A, W, G, G, G, G, G, K, K, W, P, P, W, W, W, J, J, J],
            [A, A, A, W, G, G, G, G, G, W, K, W, P, P, P, P, W, W, W, W],
            [A, A, A, W, W, W, W, G, G, W, K, W, P, P, P, P, P, P, P, P],
            [A, A, A, A, A, A, W, F, W, W, W, W, P, P, P, P, P, P, P, P],
            [A, A, A, A, A, A, W, F, F, F, F, W, P, P, P, P, P, P, P, P],
            [A, A, A, A, A, A, W, F, F, F, F, P, P, P, P, P, P, P, P, P],
            [W, W, W, B, W, W, W, W, W, W, W, W, P, W, W, W, W, W, W, W],
            [B, B, B, B, W, E, E, E, E, E, W, O, O, O, O, O, W, N, N, N],
            [B, B, B, B, W, E, E, E, E, E, W, O, O, O, O, O, O, N, N, N],
            [B, B, B, B, E, E, E, E, E, E, W, O, O, O, O, O, W, N, N, N],
            [B, B, B, B, E, W, E, E, E, W, W, W, W, W, O, O, W, N, N, N],
            [W, C, W, W, E, W, W, W, E, W, L, L, L, W, O, W, W, W, N, W],
            [C, C, C, W, W, W, D, D, D, W, L, L, L, W, W, M, M, M, M, M],
            [C, C, C, W, D, D, D, D, D, W, L, L, L, L, M, M, M, M, M, M],
            [C, C, C, W, W, W, D, D, D, L, L, L, L, L, W, M, M, M, M, M],
        ]
        room_layout = np.array(room_layout)
        agent_pos = (19, 0)
        flags_pos = [(0, 1), (2, 18), (5, 6)]
        goal_pos = (15, 0)
        # self.goal = (4, 19)
    elif maze_name == "low_connectivity2":
        room_layout = [
            [A, A, A, W, H, H, H, I, I, I, I, I, I, I, W, J, J, J, J, J],  ##Low Connectivity
            [A, A, A, W, H, H, H, W, I, I, I, I, I, I, J, J, J, J, J, J],
            [A, A, A, W, H, H, H, W, I, I, I, I, I, I, W, J, J, J, J, J],
            [A, A, A, W, W, W, H, W, W, W, K, W, W, W, W, J, J, J, J, J],
            [A, A, A, W, G, G, G, G, G, W, K, W, P, P, W, J, J, J, J, J],
            [A, A, A, W, G, G, G, G, G, W, K, W, P, P, W, W, W, J, J, J],
            [A, A, A, W, G, G, G, G, G, W, K, W, P, P, P, P, W, W, W, W],
            [A, A, A, W, W, W, W, G, G, W, K, W, P, P, P, P, P, P, P, P],
            [A, A, A, A, A, A, W, F, W, W, W, W, P, P, P, P, P, P, P, P],
            [A, A, A, A, A, A, W, F, F, F, F, W, P, P, P, P, P, P, P, P],
            [A, A, A, A, A, A, Z, F, F, F, F, P, P, P, P, P, P, P, P, P],
            [W, W, W, B, W, W, W, W, W, W, W, W, P, W, W, W, W, W, W, W],
            [B, B, B, B, W, E, E, E, E, E, W, O, O, O, O, O, W, N, N, N],
            [B, B, B, B, W, E, E, E, E, E, W, O, O, O, O, O, O, N, N, N],
            [B, B, B, B, E, E, E, E, E, E, W, O, O, O, O, O, W, N, N, N],
            [B, B, B, B, E, W, E, E, E, W, W, W, W, W, O, O, W, N, N, N],
            [W, C, W, W, E, W, W, W, E, W, L, L, L, W, O, W, W, W, N, W],
            [C, C, C, W, W, W, D, D, D, W, L, L, L, W, W, M, M, M, M, M],
            [C, C, C, W, D, D, D, D, D, W, L, L, L, L, M, M, M, M, M, M],
            [C, C, C, W, W, W, D, D, D, L, L, L, L, L, W, M, M, M, M, M],
        ]
        room_layout = np.array(room_layout)
        agent_pos = (19, 0)
        flags_pos = [(0, 1), (2, 18), (5, 6)]
        goal_pos = (15, 0)
        # self.goal = (4, 19)
    elif maze_name == "low_connectivity3":
        room_layout = [
            [A, A, A, W, H, H, H, Z, I, I, I, I, I, I, W, J, J, J, J, J],  ##Low Connectivity3
            [A, A, A, W, H, H, H, W, I, I, I, I, I, I, Z, J, J, J, J, J],
            [A, A, A, W, H, H, H, W, I, I, I, I, I, I, W, J, J, J, J, J],
            [A, A, A, W, W, W, Z, W, W, W, K, W, W, W, W, J, J, J, J, J],
            [A, A, A, W, G, G, G, G, G, W, K, W, P, P, W, J, J, J, J, J],
            [A, A, A, W, G, G, G, G, G, W, K, W, P, P, W, W, W, J, J, J],
            [A, A, A, W, G, G, G, G, G, W, K, W, P, P, P, P, W, W, W, W],
            [A, A, A, W, W, W, W, G, G, W, K, W, P, P, P, P, P, P, P, P],
            [A, A, A, A, A, A, W, Z, W, W, W, W, P, P, P, P, P, P, P, P],
            [A, A, A, A, A, A, W, F, F, F, F, W, P, P, P, P, P, P, P, P],
            [A, A, A, A, A, A, W, F, F, F, F, Z, P, P, P, P, P, P, P, P],
            [W, W, W, Z, W, W, W, W, W, W, W, W, Z, W, W, W, W, W, W, W],
            [B, B, B, B, W, E, E, E, E, E, W, O, O, O, O, O, W, N, N, N],
            [B, B, B, B, W, E, E, E, E, E, W, O, O, O, O, O, Z, N, N, N],
            [B, B, B, B, Z, E, E, E, E, E, W, O, O, O, O, O, W, N, N, N],
            [B, B, B, B, B, W, E, E, E, W, W, W, W, W, O, O, W, N, N, N],
            [W, Z, W, W, B, W, W, W, Z, W, L, L, L, W, O, W, W, W, Z, W],
            [C, C, C, W, W, W, D, D, D, W, L, L, L, W, W, M, M, M, M, M],
            [C, C, C, W, D, D, D, D, D, W, L, L, L, L, Z, M, M, M, M, M],
            [C, C, C, W, W, W, D, D, D, Z, L, L, L, L, W, M, M, M, M, M],
        ]
        room_layout = np.array(room_layout)
        agent_pos = (19, 0)
        flags_pos = [(0, 1), (2, 18), (5, 6)]
        goal_pos = (15, 0)
        # self.goal = (4, 19)
    elif maze_name.startswith("external"):
        path = f"/workspace/masterthesis/external_mazes/{maze_name}.txt"
        room_layout = []
        with open(path, "r") as f:
            content = f.readlines()
            content = [x.strip() for x in content]
            for item in content:
                item = item.replace("#", "w")
                # item = item.replace('.', '0')
                row = [x for x in item]
                room_layout.append(row)
        room_layout = np.array(room_layout)
    else:
        raise Exception("invalide maze name")
    return room_layout, agent_pos, flags_pos, goal_pos
