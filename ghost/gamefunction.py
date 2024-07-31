# -*- coding: utf-8 -*-
"""
万妖行游戏运行算法代码
:Author: bo
"""
import os

import cv2
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal,QThread


class ToolsSignals(QObject):
    blocks_updated = pyqtSignal(np.ndarray, int)



class GameFunction(QObject):
    def __init__(self):
        super().__init__()
        self.pictureSize = 94
        self.fig_x = [31, 131, 232, 333, 435]
        self.fig_y = [31, 131, 232, 333, 435]
        self.ghost = {i.split('.')[0]: cv2.imread(f'ghost/{i}').astype(int)[:, :, [2, 1, 0]] for i in
                      os.listdir('ghost')}
        self.signals = ToolsSignals()
        # ##怪物战斗力对应表
        self.ghostforce_dict = {
            1: 100,
            2: 310,
            3: 640,
            4: 1770,
            5: 3540,
            6: 6140,
            7: 9790,
            8: 14770,
            9: 21410,
            10: 33140,
            11: 49540,
            12: 72300,
            13: 103730,
            14: 147030,
            15: 206630,
            16: 288720,
            17: 401950,
            18: 558510,
            19: 775650,
            20: 1077880
        }

        # #怪物等级随等级的限制
        self.level_limits_dict = {
            1: [1, 7],
            2: [1, 8],
            3: [2, 9],
            4: [3, 10],
            5: [4, 11],
            6: [5, 12],
            7: [6, 13],
            8: [7, 14],
            9: [8, 15],
            10: [9, 16],
            11: [10, 17],
            12: [11, 18],
            13: [12, 19],
            14: [13, 20],
            15: [14, 20]
        }

        # #每一轮步数与等级的对应表
        self.level_steps_dict = {
            1: 2,
            2: 6,
            3: 6,
            4: 7,
            5: 7,
            6: 8,
            7: 8,
            8: 8,
            9: 9,
            10: 9,
            11: 9,
            12: 10,
            13: 10,
            14: 10,
            15: 10,
        }

        # #每一轮体力与等级对应表
        self.level_cost_dict = {
            1: 1,
            2: 1,
            3: 2,
            4: 2,
            5: 3,
            6: 4,
            7: 5,
            8: 6,
            9: 7,
            10: 8,
            11: 8,
            12: 9,
            13: 9,
            14: 10,
            15: 10,
        }

        # #棋盘格
        self.blocks = np.array([[1, 2, 4, 5, 2],
                                [3, 3, 1, 3, 1],
                                [1, 1, 2, 2, 3],
                                [2, 3, 1, 4, 3],
                                [1, 2, 1, 2, 2]])
        self.score = 0

    ####
    def get_game_blocks(self):
        return self.blocks

    # 更新棋盘格
    def update_blocks(self, new_block):
        self.blocks = new_block

    def update_score(self, score):
        self.score = score

    # 图像识别方法
    def recognize(self, figure):
        blocks = np.array([[figure[self.fig_x[i]:self.fig_x[i] + self.pictureSize, :][:,
                            self.fig_y[j]:self.fig_y[j] + self.pictureSize] for j in range(5)] for i in range(5)],
                          dtype=int)
        blocks = blocks.reshape((5, 5, 1, self.pictureSize, self.pictureSize, 3)).repeat(len(self.ghost), axis=2)
        return np.array([int(key.split('-')[0]) for key in self.ghost.keys()])[
            ((blocks - np.array(list(self.ghost.values()))) ** 2).sum(axis=(-1, -2, -3)).argmin(axis=2)]

    def get_blocks_score(self, blocks, limit):
        blank = list(range(limit[0], limit[1] - 2)) * 2 + [limit[1] - 2]
        new_force = self.ghostforce_dict.copy()
        new_force[0] = np.mean([new_force[i] for i in blank])
        return np.sum([new_force[block] for block in blocks.ravel()])

    def same_neighbour(self, blocks, loc):
        outlocs = [loc]
        locs = [loc]
        while len(locs) > 0:
            x, y = locs.pop()
            for i, j in [[x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]]:
                if i < 0 or i >= 5 or j < 0 or j >= 5:
                    continue
                if blocks[i, j] == blocks[x, y] and [i, j] not in outlocs:
                    outlocs.append([i, j])
                    locs.append([i, j])
        return np.array(outlocs)

    def pitch(self, blocks, loc):
        new_blocks = blocks.copy()
        new_blocks[(*loc,)] = 0
        return new_blocks, self.ghostforce_dict[blocks[(*loc,)]]

    def upgrade(self, blocks, loc):
        new_blocks = blocks.copy()
        new_blocks[(*loc,)] += 1
        return new_blocks, 0

    def synthesize(self, blocks, limit, upgrade_loc=None):
        new_blocks = blocks.copy()

        if upgrade_loc is not None:
            x, y = upgrade_loc
            same_neighb = self.same_neighbour(new_blocks, upgrade_loc)
            if same_neighb.shape[0] < 3:
                return new_blocks, 0
            new_blocks[(*same_neighb.T,)] = 0
            new_blocks[x, y] = min(limit[1], blocks[x, y] + same_neighb.shape[0] - 2)
            return new_blocks, self.ghostforce_dict[blocks[x, y]] * (same_neighb.shape[0] - 2)

        same_groups = []
        flag = np.zeros(blocks.shape, dtype=int)
        for x in range(5):
            for y in range(5):
                if new_blocks[x, y] == 0 or flag[x, y] == 1:
                    continue
                same_neighb = self.same_neighbour(new_blocks, [x, y])
                if same_neighb.shape[0] < 3:
                    continue
                flag[(*same_neighb.T,)] = 1
                same_groups.append(same_neighb)
        if len(same_groups) == 0:
            return new_blocks, 0
        score = 0
        for group in same_groups:
            new_blocks[(*group.T,)] = 0
            x, y = sorted(group, key=lambda a: (-a[0], a[1]))[0]
            new_blocks[x, y] = min(limit[1], blocks[x, y] + group.shape[0] - 2)
            score += self.ghostforce_dict[blocks[x, y]] * (group.shape[0] - 2)
        return new_blocks, score

    def fill_random(self, blocks, limit):
        new_blocks = blocks.copy()
        zeros = np.argwhere(new_blocks == 0)
        for loc in zeros:
            new_blocks[tuple(loc)] = np.random.randint(limit[0], limit[1]-2)
        return new_blocks

    def move(self, blocks, limit):
        new_blocks = blocks.copy()
        reach_limit = np.where(new_blocks == limit[1])
        new_blocks[reach_limit] = 0
        score = self.ghostforce_dict[limit[1]] * reach_limit[0].shape[0]

        for j in range(5):
            for i in range(4):
                if new_blocks[i + 1, j] == 0:
                    new_blocks[1:i + 2, j] = new_blocks[:i + 1, j]
                    new_blocks[0, j] = 0

        return new_blocks, score

    def action_with_random(self, blocks, op, loc, limit, score_blocks=False):
        new_blocks, score = op(blocks, loc)
        if op == self.upgrade:
            new_blocks, s = self.synthesize(new_blocks, limit, loc)
            score += s

        while True:
            new_blocks, s = self.move(new_blocks, limit)
            score += s
            # self.signals.blocks_updated.emit(new_blocks, score)
            # QThread.sleep(100)
            new_blocks, s = self.synthesize(new_blocks, limit)
            score += s

            # self.signals.blocks_updated.emit(new_blocks, score)
            # QThread.sleep(100)
            new_blocks = self.fill_random(new_blocks, limit)

            new_blocks, s = self.synthesize(new_blocks, limit)
            score += s
            # self.signals.blocks_updated.emit(new_blocks, score)
            # QThread.sleep(100)
            if s == 0:
                if score_blocks:
                    score += self.get_blocks_score(new_blocks, limit) - self.get_blocks_score(blocks, limit)
                return new_blocks, score

    def action(self, blocks, op, loc, limit, score_blocks=False):
        new_blocks, score = op(blocks, loc)
        if op == self.upgrade:
            new_blocks, s = self.synthesize(new_blocks, limit, loc)
            score += s

        while True:
            new_blocks, s = self.move(new_blocks, limit)
            score += s
            new_blocks, s = self.synthesize(new_blocks, limit)
            score += s
            if s == 0:
                if score_blocks:
                    score += self.get_blocks_score(new_blocks, limit) - self.get_blocks_score(blocks, limit)
                return new_blocks, score

    def optimize(self, blocks, limit, steps, results=3, score_blocks=False):
        if steps == 0:
            return None, [0]

        op = [self.pitch, self.upgrade]
        scores = -np.ones((5, 5, 2), dtype=int)
        highest = blocks.max()
        for i in range(5):
            for j in range(5):
                if blocks[i, j] == 0:
                    continue
                for k in range(2):
                    if blocks[i, j] == highest and k == 1:
                        continue
                    new_blocks, scores[i, j, k] = self.action(blocks, op[k], [i, j], limit, score_blocks)
                    scores[i, j, k] += max(self.optimize(new_blocks, limit, steps - 1, results, score_blocks)[1])
        max_ind = np.unravel_index(scores.argsort(axis=None)[-results:][::-1], scores.shape)
        return np.array(max_ind).T, scores[max_ind]

    operation = ["上阵", "点击"]

    def printf_optimal_move(self, optimal_moves, scores):
        for i in range(optimal_moves.shape[0]):
            row, col, op_type = optimal_moves[i]
            score = scores[i]
            if op_type != -1:
                operate = self.operation[op_type]
                print(f"{operate}第{row + 1}行第{col + 1}列，能加大概{score}分")
