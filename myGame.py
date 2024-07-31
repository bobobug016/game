import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QLabel, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap,QPainter, QColor,QMouseEvent
from PyQt5.QtCore import Qt, QPoint,QRect,QThread,pyqtSignal, pyqtSlot, QObject, QMutex
import numpy as np
import gamefunction

class WorkerThread(QThread):
    update_signal = pyqtSignal(np.ndarray, int)

    def __init__(self, method, *args, **kwargs):
        super().__init__()
        self.method = method
        self.args = args
        self.kwargs = kwargs

    def run(self):
        self.method(*self.args, **self.kwargs)

class FrameWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("border: 3px solid gray;border-radius:2px")  # 设置方框的边框

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QColor(0, 0, 0))  # 设置边框颜色
        painter.drawRect(self.rect())  # 绘制方框

class GameWindow(QMainWindow):
    update_signal = pyqtSignal(np.ndarray, int)
    def __init__(self):
        super().__init__()
        self.game = gamefunction.GameFunction()
        self.myMatrix = self.game.blocks.copy()
        self.score = 28120
        self.level = 2  # 当前等级
        self.limit = self.game.level_limits_dict[self.level]
        self.drag_start_position = None
        self.picked_position = None
        self.is_dragging = False  # 添加拖拽标志
        self.mutex = QMutex()
        # self.pick_station = [-1,-1]

        # 设置窗口标题和大小
        self.setWindowTitle("万妖行")
        self.setGeometry(100, 100, 600, 850)

        # 初始化UI
        self.init_ui()
        # self.initWorker()

        self.update_signal.connect(self.update_blocks)

        self.update_limit_label()
        # 显示窗口
        self.show()

    def init_ui(self):
        # 创建中央小部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        # 创建得分区部件
        self.score_area = QWidget(central_widget)
        self.score_area.setGeometry(10,10,580,250)  # 固定得分区的高度
        self.score_area.setStyleSheet("background-color: lightgray; border: 1px solid black;")  # 设置背景颜色和边框
        # 创建顶部布局，包含分数和等级显示
        self.init_top_layout()

        # 创建方框部件和网格布局
        self.frame_widget = FrameWidget(central_widget)
        self.frame_widget.setGeometry(20,270,560,560)
        # 初始化网格中的标签或图片
        self.init_grid_labels()
        self.update_grid_images(self.myMatrix)

        self.score_area_rect = QRect(self.score_area.geometry())



    def update_limit_label(self):
        # 更新显示等级和 limit
        self.level_label.setText(f"Level: {self.level}")
        # self.limit_label.setText(f"Limit: {self.limit}")

    def init_top_layout(self):
        self.score_label = QLabel("Score: 0",self.score_area)
        self.score_label.setGeometry(320,80,200,40)
        self.level_label = QLabel("Level: 1",self.score_area)
        self.level_label.setGeometry(20, 80, 200, 40)


    def init_grid_labels(self):
        cell_size = 100  # 设置每个单元格的大小
        self.grid = [[QLabel(self.frame_widget) for _ in range(5)] for _ in range(5)]
        for i in range(5):
            for j in range(5):
                # self.grid[i][j].setFixedSize(cell_size, cell_size)  # 设置固定大小
                self.grid[i][j].setGeometry(10+j*110,10+i*110,cell_size,cell_size)
                self.grid[i][j].setStyleSheet("text-align:center")
                # self.grid[i][j].mousePressEvent = self.handle_label_click  # 绑定点击事件

    def handle_label_click(self, event: QMouseEvent):
        # 获取被点击的 QLabel
        widget = self.sender()
        if widget:
            self.picked_position = self.get_grid_position(widget)
            self.drag_start_position = event.pos()
            self.is_dragging = False

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            widget = self.childAt(event.pos())
            if widget and isinstance(widget, QLabel):
                self.picked_position = self.get_grid_position(widget)
                self.drag_start_position = event.pos()
                self.is_dragging = False

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() & Qt.LeftButton and self.picked_position:
            if self.drag_start_position and (event.pos() - self.drag_start_position).manhattanLength() > 5:
                self.is_dragging = True

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self.picked_position:
            # 处理拖拽释放位置
            # 更新得分区的矩形区域
            self.score_area_rect = QRect(self.score_area.geometry())
            if self.is_dragging:
                if self.score_area_rect.contains(event.pos()):
                # 鼠标释放在得分区内，执行 pitch 操作
                    widget = self.childAt(event.pos())
                    grid_pos = self.get_grid_position(widget)
                    if grid_pos is None:
                        self.action(self.game.pitch, *self.picked_position)
            else:
                self.action(self.game.upgrade, *self.picked_position)
                # 执行 pitch 操作
            self.picked_position = None
            self.is_dragging = False  # 重置拖拽标志
    def get_grid_position(self, widget):
        # 计算鼠标位置对应的网格位置
        for i in range(5):
            for j in range(5):
                if self.grid[i][j] == widget:
                    return (i, j)
        return None

    def action_step_with_random(self, blocks, op, loc, limit, score_blocks=False):
        new_blocks, score = op(blocks, loc)
        if op == self.game.upgrade:
            new_blocks, s = self.game.synthesize(new_blocks, limit, loc)
            score += s
            self.update_signal.emit(new_blocks, s)
            # QThread.sleep(100)

        while True:
            new_blocks, s = self.game.move(new_blocks, limit)
            score += s
            # self.signals.blocks_updated.emit(new_blocks, score)
            # QThread.sleep(100)
            new_blocks, s = self.game.synthesize(new_blocks, limit)
            score += s
            self.update_signal.emit(new_blocks, s)
            # QThread.sleep(100)

            # self.signals.blocks_updated.emit(new_blocks, score)
            # QThread.sleep(100)
            new_blocks = self.game.fill_random(new_blocks, limit)
            self.update_signal.emit(new_blocks, s)
            # QThread.sleep(100)

            new_blocks, s = self.game.synthesize(new_blocks, limit)
            score += s
            self.update_signal.emit(new_blocks, s)
            # QThread.sleep(100)
            # self.signals.blocks_updated.emit(new_blocks, score)
            # QThread.sleep(100)
            if s == 0:
                if score_blocks:
                    score += self.game.get_blocks_score(new_blocks, limit) - self.game.get_blocks_score(blocks, limit)
                    self.myMatrix = new_blocks
                    self.score += score
                return new_blocks, score

    def start_thread(self,blocks,op,loc):
        self.thread = WorkerThread(self.action_step_with_random, blocks, op, loc, limit=self.limit)
        self.update_signal.connect(self.update_ui)
        self.thread.start()

    def action(self, op, *args):
        # 执行操作并更新块数据和分数
        if len(args) == 2:
            x,y =args[0],args[1]
            # self.myMatrix, score = self.game.action_with_random(self.myMatrix, op, [x,y], limit=self.limit)
            self.myMatrix, score = self.action_step_with_random(self.myMatrix, op, [x,y], limit=self.limit)
            self.score += score

        self.update_blocks(self.myMatrix,self.score)

    def update_blocks(self,blocks, score):
        # 更新网格并显示分数
        self.update_grid_images(blocks)
        self.score_label.setText(f"Score: {self.score}")

    def level_up(self):
        if self.level < max(self.game.level_limits_dict.keys()):
            self.level += 1
            self.limit = self.game.level_limits_dict[self.level]
            self.update_limit_label()

    def update_grid_images(self, blocks):
        for i in range(5):
            for j in range(5):
                number = blocks[i][j]
                if number > 0:
                    pixmap = QPixmap(os.path.join('ghost', f'{number}.png'))
                    pixmap = pixmap.scaled(100, 100)  # 将图片缩放到固定大小
                    self.grid[i][j].setPixmap(pixmap)
                    # self.grid[i][j].setText(f"{number}")
                else:
                    self.grid[i][j].clear()

    def closeEvent(self, event):
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GameWindow()
    sys.exit(app.exec_())
