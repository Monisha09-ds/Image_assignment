import numpy as np

class KalmanFilter:
    def __init__(self, initial_state, dt=1.0):
        """
        Initializes the Kalman Filter.
        State vector x = [px, py, vx, vy]^T
        """
        # State vector
        self.x = np.array(initial_state).reshape(-1, 1).astype(float)
        
        # State transition matrix
        self.A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix (we only measure position)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Covariance matrix
        self.P = np.eye(4) * 10.0
        
        # Process noise covariance
        self.Q = np.eye(4) * 0.1
        
        # Measurement noise covariance
        self.R = np.eye(2) * 1.0
        
        # Identity matrix
        self.I = np.eye(4)
        
        # History for visualization
        self.history = [self.x[:2].copy()]

    def predict(self):
        """
        Predicts the next state.
        """
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x[:2]

    def update(self, y):
        """
        Updates the state based on a measurement y = [px, py]^T.
        """
        y = np.array(y).reshape(-1, 1).astype(float)
        
        # Innovation
        innovation = y - (self.H @ self.x)
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman Gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        self.x = self.x + K @ innovation
        self.P = (self.I - K @ self.H) @ self.P
        
        self.history.append(self.x[:2].copy())
        return self.x[:2]

    def get_position(self):
        return self.x[:2].flatten()
