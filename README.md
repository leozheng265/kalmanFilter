# Kalman Filter
The KalmanFilter class implements a standard Kalman filter algorithm to estimate the state of a linear system. The filter can handle control input and multiple measurements from multiple sensors.

## Requirements
* Python 3.6+
* NumPy library

## Virtual Environment Set Up
1. Create a virtual environment:
    * `python3 -m venv myenv` (replace myenv with name of your virtual environment)
2. Activate the virtual environment:
    * On Linux: `source myenv/bin/activate`
    * On Windows: `myenv\Scripts\activate`
3. Install the dependencies:
    * `pip install -r requirements.txt`
4. Deactivate the virtual environment:
    * `deactivate`

## Input
* **A** - State transition matrix (n x n)
* **B** - Control matrix (n x m)
* **H** - Measurement matrix (m x n)
* **x**: The initial state estimate (n x 1)
* **P**: The initial estimate covariance (n x n)
* **Q**: The process noise covariance (n x n)
* **R**: The measurement noise covariance (m x m)

## Output
* **x**: The state estimate (n x 1)

## Methods
### `predict`

The `predict` method predicts the next state based on the current state and control input.

Input:
* **u**: The control input. Defaults to none (m x 1).

Output:
* **x**: The predicted state (n x 1).

### `update`

The `update` method updates the Kalman filter using the latest measurements from all sensors.

Input:
* `measurements`: List of ndarrays, where each ndarray represents the measurement from one sensor.

### `validate_input`

The `validate_input` method validates the input matrices and vectors, checking if they are numpy arrays and if they have the correct shape.
