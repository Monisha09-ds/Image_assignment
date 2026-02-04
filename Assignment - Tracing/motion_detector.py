import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import dilation, square
from kalman_filter import KalmanFilter
from scipy.spatial.distance import euclidean

class TrackedObject:
    def __init__(self, obj_id, initial_pos, alpha):
        self.id = obj_id
        self.kf = KalmanFilter([initial_pos[0], initial_pos[1], 0, 0])
        self.frames_active = 1
        self.frames_inactive = 0
        self.is_confirmed = False
        self.alpha = alpha
        self.last_measurement = initial_pos

    def update_measurement(self, pos):
        self.kf.update(pos)
        self.frames_active += 1
        self.frames_inactive = 0
        self.last_measurement = pos
        if self.frames_active >= self.alpha:
            self.is_confirmed = True

    def update_miss(self):
        self.kf.predict()
        self.frames_inactive += 1
        # If we miss a measurement, we don't increment frames_active but we don't reset it either?
        # Conventional logic: frames_active is for consecutive hits for confirmation.
        if not self.is_confirmed:
            self.frames_active = 0 

class MotionDetector:
    def __init__(self, alpha=3, tau=25, delta=50, s=1, N=10):
        """
        alpha: frame hysteresis
        tau: motion threshold
        delta: distance threshold
        s: frames to skip (not used directly in this class but in the loop)
        N: max objects
        """
        self.alpha = alpha
        self.tau = tau
        self.delta = delta
        self.N = N
        
        self.prev_frame = None
        self.prev_prev_frame = None
        
        self.tracked_objects = []
        self.next_obj_id = 0
        self.candidates = [] # Objects not yet confirmed

    def process_frame(self, frame):
        """
        frame: grayscale frame
        """
        if self.prev_frame is None:
            self.prev_frame = frame
            return []
            
        if self.prev_prev_frame is None:
            self.prev_prev_frame = self.prev_frame
            self.prev_frame = frame
            return []
            
        # 1. Motion Detection
        d1 = np.abs(frame.astype(float) - self.prev_frame.astype(float))
        d2 = np.abs(self.prev_frame.astype(float) - self.prev_prev_frame.astype(float))
        motion = np.minimum(d1, d2)
        
        # 2. Thresholding
        thresh = (motion > self.tau).astype(np.uint8)
        
        # 3. Dilation
        dilated = dilation(thresh, square(9))
        
        # 4. Connected Components
        labels = label(dilated)
        regions = regionprops(labels)
        
        measurements = []
        for prop in regions:
            if prop.area < 20: # Filter noise by size
                continue
            # Centroid is (row, col)
            measurements.append((prop.centroid[1], prop.centroid[0]))
            
        # 5. Data Association
        self._associate_and_update(measurements)
        
        # Update frame buffer
        self.prev_prev_frame = self.prev_frame
        self.prev_frame = frame
        
        return self.get_active_tracks()

    def _associate_and_update(self, measurements):
        # Predict all existing objects
        for obj in self.tracked_objects:
            obj.kf.predict()
            
        unassigned_measurements = list(measurements)
        assigned_objs = set()

        # Simple Greedy Association
        for obj in self.tracked_objects:
            if not unassigned_measurements:
                break
            
            pred_pos = obj.kf.get_position()
            # Find closest measurement
            best_idx = -1
            min_dist = float('inf')
            
            for i, m_pos in enumerate(unassigned_measurements):
                dist = euclidean(pred_pos, m_pos)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
            
            if min_dist < self.delta:
                obj.update_measurement(unassigned_measurements.pop(best_idx))
                assigned_objs.add(obj)

        # Handle misses
        for obj in self.tracked_objects:
            if obj not in assigned_objs:
                obj.update_miss()

        # Handle new tracks
        for m_pos in unassigned_measurements:
            if len(self.tracked_objects) < self.N:
                new_obj = TrackedObject(self.next_obj_id, m_pos, self.alpha)
                self.tracked_objects.append(new_obj)
                self.next_obj_id += 1

        # Cleanup inactive objects
        self.tracked_objects = [obj for obj in self.tracked_objects 
                               if obj.frames_inactive < self.alpha]

    def get_active_tracks(self):
        # Only return objects that have been confirmed (active over alpha frames)
        return [obj for obj in self.tracked_objects if obj.is_confirmed]

    def reset(self):
        self.tracked_objects = []
        self.prev_frame = None
        self.prev_prev_frame = None
