NUM_ROBOTS = 4

VX_Sigma = 0.15
VY_Sigma = 0.15
VA_sigma = 0.06

SENSOR_VAR_X = VX_Sigma * VX_Sigma
SENSOR_VAR_Y = VY_Sigma * VY_Sigma
ORIENTATION_VAR = VA_sigma * VA_sigma

MType = 'pos' 

P_Sigma = 0.1
POS_VAR = P_Sigma * P_Sigma

RANGE_DISTURB = 0.0

# prob = 0.24

prob = 0.24

ITER_NUM = 100

STEP = 2.0
DURATION = 240

DATA_RECORDER = False

# TIME_MARK = '2023-10-13 13:56:18'
# TIME_MARK = '2024-04-13 14:57:46' # prob = 0.3 disturb = 0.3
# TIME_MARK = '2024-04-13 20:36:41' # prob = 0.2
# TIME_MARK = '2024-04-13 20:40:07' # prob = 0.3
# TIME_MARK = '2024-04-13 23:16:27' # prob = 0.24 RAV V1
# TIME_MARK = '2024-09-05 13:18:32' # prob = 0.24 RAV V2 0.08 rad/s
# TIME_MARK = '2024-09-05 13:47:38' # prob = 0.24 RAV V2 0.06 rad/s 240s
# TIME_MARK = '2024-09-05 15:30:27' # prob = 0.24 RAV V2 0.06 rad/s 240s
# TIME_MARK = '2024-09-22 14:49:56' # prob = 1.0
# TIME_MARK = '2024-09-22 16:03:50' # prob = 0.5
TIME_MARK = '2024-09-22 16:08:27' # prob = 0.3
# TIME_MARK = '2024-05-10 09:38:23' # doc 1
# TIME_MARK = '2024-05-10 12:06:54' # doc 2
# TIME_MARK = '2024-05-12 16:39:25' # doc 3
# TIME_MARK = '2024-05-12 16:41:20'
