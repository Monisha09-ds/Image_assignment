import sys
import numpy as np
import argparse
from PySide6 import QtCore, QtWidgets, QtGui
from skvideo.io import vread
from motion_detector import MotionDetector

# Fix for some skvideo/numpy versions
np.float = np.float64
np.int = np.int_

class TrackerApp(QtWidgets.QWidget):
    def __init__(self, frames):
        super().__init__()

        self.frames = frames
        self.current_frame = 0
        
        # Initialize Motion Detector
        self.detector = MotionDetector(alpha=3, tau=25, delta=60, N=15)
        self.tracking_results = {} # Cache for results if needed

        # UI Components
        self.img_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.img_label.setMinimumSize(800, 600)
        
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(self.frames.shape[0]-1)
        self.frame_slider.setValue(0)

        # Buttons
        self.btn_prev = QtWidgets.QPushButton("-1")
        self.btn_next = QtWidgets.QPushButton("+1")
        self.btn_back_60 = QtWidgets.QPushButton("-60")
        self.btn_fwd_60 = QtWidgets.QPushButton("+60")
        
        # Layouts
        ctrl_layout = QtWidgets.QHBoxLayout()
        ctrl_layout.addWidget(self.btn_back_60)
        ctrl_layout.addWidget(self.btn_prev)
        ctrl_layout.addWidget(self.btn_next)
        ctrl_layout.addWidget(self.btn_fwd_60)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.img_label)
        layout.addWidget(self.frame_slider)
        layout.addLayout(ctrl_layout)

        # Connections
        self.frame_slider.sliderMoved.connect(self.on_slider_move)
        self.btn_next.clicked.connect(lambda: self.update_frame_delta(1))
        self.btn_prev.clicked.connect(lambda: self.update_frame_delta(-1))
        self.btn_fwd_60.clicked.connect(lambda: self.update_frame_delta(60))
        self.btn_back_60.clicked.connect(lambda: self.update_frame_delta(-60))

        self.update_display()

    def update_frame_delta(self, delta):
        new_frame = np.clip(self.current_frame + delta, 0, self.frames.shape[0]-1)
        if delta < 0 or new_frame < self.current_frame:
            # Re-initialize for backward motion
            self.detector.reset()
            self.current_frame = 0 # Need to re-process from 0 to target?
            # Actually simplest for the assignment: "If user slides back... model should re-initialize"
            # We'll just start from the new frame as if it's the start.
            self.current_frame = new_frame
        else:
            # Step forward and process intermediate frames if skipped
            while self.current_frame < new_frame:
                self.current_frame += 1
                frame = self.get_gray_frame(self.current_frame)
                self.detector.process_frame(frame)
        
        self.frame_slider.setValue(self.current_frame)
        self.update_display()

    def on_slider_move(self, pos):
        if pos < self.current_frame:
            self.detector.reset()
        
        # For slider, we might need to process a lot, let's just jump and reset for now
        # as processing every frame during a big slide is too slow.
        self.detector.reset()
        self.current_frame = pos
        self.update_display()

    def get_gray_frame(self, idx):
        frame = self.frames[idx]
        if len(frame.shape) == 3:
            # Simple average or luminance
            return np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        return frame

    def update_display(self):
        frame = self.frames[self.current_frame].copy()
        h, w = frame.shape[:2]
        
        # Process frame to get tracks
        gray = self.get_gray_frame(self.current_frame)
        tracks = self.detector.process_frame(gray)
        
        # Draw tracks on the copy
        self.draw_tracks(frame, tracks)
        
        # Convert to QImage
        if len(frame.shape) == 3:
            q_img = QtGui.QImage(frame.data, w, h, w * 3, QtGui.QImage.Format_RGB888)
        else:
            q_img = QtGui.QImage(frame.data, w, h, w, QtGui.QImage.Format_Grayscale8)
            
        self.img_label.setPixmap(QtGui.QPixmap.fromImage(q_img))

    def draw_tracks(self, frame, tracks):
        painter = QtGui.QPainter()
        # This is tricky because we are drawing on a numpy array via Pixmap usually
        # Let's use OpenCV for drawing or just draw on the Pixmap after conversion.
        pass

    # Improved update_display to use QPainter on the pixmap
    def update_display(self):
        orig_frame = self.frames[self.current_frame]
        h, w = orig_frame.shape[:2]
        
        gray = self.get_gray_frame(self.current_frame)
        tracks = self.detector.process_frame(gray)
        
        if len(orig_frame.shape) == 3:
            q_img = QtGui.QImage(orig_frame.tobytes(), w, h, w * 3, QtGui.QImage.Format_RGB888)
        else:
            q_img = QtGui.QImage(orig_frame.tobytes(), w, h, w, QtGui.QImage.Format_Grayscale8)
            
        pixmap = QtGui.QPixmap.fromImage(q_img)
        
        # Draw overlay on pixmap
        painter = QtGui.QPainter(pixmap)
        for obj in tracks:
            # Draw trail
            if len(obj.kf.history) > 1:
                painter.setPen(QtGui.QPen(QtCore.Qt.green, 2))
                for i in range(len(obj.kf.history)-1):
                    p1 = obj.kf.history[i]
                    p2 = obj.kf.history[i+1]
                    painter.drawLine(p1[0], p1[1], p2[0], p2[1])
            
            # Draw current position/marker
            curr_pos = obj.kf.get_position()
            painter.setPen(QtGui.QPen(QtCore.Qt.red, 3))
            painter.drawEllipse(QtCore.Qt.Point(curr_pos[0], curr_pos[1]), 5, 5)
            painter.drawText(curr_pos[0]+10, curr_pos[1], f"ID: {obj.id}")
            
        painter.end()
        self.img_label.setPixmap(pixmap)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    args = parser.parse_args()

    frames = vread(args.video_path)
    app = QtWidgets.QApplication(sys.argv)
    window = TrackerApp(frames)
    window.show()
    sys.exit(app.exec())
