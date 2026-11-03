I use a conservative fixed-column strategy: retain the initial left-facing orientation at column 5, fire by default, and change rows to intercept divers or shoot enemies approaching from the left.

The first layer computes directional, distance-weighted row attractions, nearby hazards, player-row indicators, and gauge flags. The second layer combines these into movement preferences, destination-safety penalties, and a fire/hold preference for valuable current rows. Low oxygen with an indicated passenger triggers ascent; upper-row guards reduce accidental surfacing under sticky actions. This is an untested hand-designed heuristic, not a claim of optimality.

```edits
# First layer: upward attractions for rows 2..8, then downward attractions.
b1[0:14] = -12

# Row attractions: reward weights shared by the upward/downward pair.
W1[122:180:6, 0:14:7] = [[1.8, 1.8], [1.8, 1.8], [2, 2], [2.3, 2.3], [3.8, 3.8], [4.6, 4.6], [3.8, 3.8], [3, 3], [2.4, 2.4], [2, 2]]
W1[182:240:6, 1:14:7] = [[1.8, 1.8], [1.8, 1.8], [2, 2], [2.3, 2.3], [3.8, 3.8], [4.6, 4.6], [3.8, 3.8], [3, 3], [2.4, 2.4], [2, 2]]
W1[242:300:6, 2:14:7] = [[1.8, 1.8], [1.8, 1.8], [2, 2], [2.3, 2.3], [3.8, 3.8], [4.6, 4.6], [3.8, 3.8], [3, 3], [2.4, 2.4], [2, 2]]
W1[302:360:6, 3:14:7] = [[1.8, 1.8], [1.8, 1.8], [2, 2], [2.3, 2.3], [3.8, 3.8], [4.6, 4.6], [3.8, 3.8], [3, 3], [2.4, 2.4], [2, 2]]
W1[362:420:6, 4:14:7] = [[1.8, 1.8], [1.8, 1.8], [2, 2], [2.3, 2.3], [3.8, 3.8], [4.6, 4.6], [3.8, 3.8], [3, 3], [2.4, 2.4], [2, 2]]
W1[422:480:6, 5:14:7] = [[1.8, 1.8], [1.8, 1.8], [2, 2], [2.3, 2.3], [3.8, 3.8], [4.6, 4.6], [3.8, 3.8], [3, 3], [2.4, 2.4], [2, 2]]
W1[482:540:6, 6:14:7] = [[1.8, 1.8], [1.8, 1.8], [2, 2], [2.3, 2.3], [3.8, 3.8], [4.6, 4.6], [3.8, 3.8], [3, 3], [2.4, 2.4], [2, 2]]

# Cancel enemy attraction near and behind the submarine, retaining divers.
W1[145:180:6, 0:14:7] = [[-3.8, -3.8], [-4.6, -4.6], [-3.8, -3.8], [-3, -3], [-2.4, -2.4], [-2, -2]]
W1[205:240:6, 1:14:7] = [[-3.8, -3.8], [-4.6, -4.6], [-3.8, -3.8], [-3, -3], [-2.4, -2.4], [-2, -2]]
W1[265:300:6, 2:14:7] = [[-3.8, -3.8], [-4.6, -4.6], [-3.8, -3.8], [-3, -3], [-2.4, -2.4], [-2, -2]]
W1[325:360:6, 3:14:7] = [[-3.8, -3.8], [-4.6, -4.6], [-3.8, -3.8], [-3, -3], [-2.4, -2.4], [-2, -2]]
W1[385:420:6, 4:14:7] = [[-3.8, -3.8], [-4.6, -4.6], [-3.8, -3.8], [-3, -3], [-2.4, -2.4], [-2, -2]]
W1[445:480:6, 5:14:7] = [[-3.8, -3.8], [-4.6, -4.6], [-3.8, -3.8], [-3, -3], [-2.4, -2.4], [-2, -2]]
W1[505:540:6, 6:14:7] = [[-3.8, -3.8], [-4.6, -4.6], [-3.8, -3.8], [-3, -3], [-2.4, -2.4], [-2, -2]]

# Player-position gating and distance discounts for upward attractions.
W1[150:540:60, 0:7] = [
[10, 0, 0, 0, 0, 0, 0],
[9.4, 10, 0, 0, 0, 0, 0],
[8.8, 9.4, 10, 0, 0, 0, 0],
[8.2, 8.8, 9.4, 10, 0, 0, 0],
[7.6, 8.2, 8.8, 9.4, 10, 0, 0],
[7, 7.6, 8.2, 8.8, 9.4, 10, 0],
[6.4, 7, 7.6, 8.2, 8.8, 9.4, 10]
]

# Player-position gating and distance discounts for downward attractions.
W1[30:540:60, 7:14] = [
[8.8, 8.2, 7.6, 7, 6.4, 5.8, 5.2],
[9.4, 8.8, 8.2, 7.6, 7, 6.4, 5.8],
[10, 9.4, 8.8, 8.2, 7.6, 7, 6.4],
[0, 10, 9.4, 8.8, 8.2, 7.6, 7],
[0, 0, 10, 9.4, 8.8, 8.2, 7.6],
[0, 0, 0, 10, 9.4, 8.8, 8.2],
[0, 0, 0, 0, 10, 9.4, 8.8],
[0, 0, 0, 0, 0, 10, 9.4],
[0, 0, 0, 0, 0, 0, 10]
]

# Hazard summaries for rows 1..8.
b1[14:22] = -3
W1[61:120:6, 14] = [0, 0, 0, 3, 6, 7, 6, 3, 0, 0]
W1[121:180:6, 15] = [0, 0, 0, 3, 6, 7, 6, 3, 0, 0]
W1[181:240:6, 16] = [0, 0, 0, 3, 6, 7, 6, 3, 0, 0]
W1[241:300:6, 17] = [0, 0, 0, 3, 6, 7, 6, 3, 0, 0]
W1[301:360:6, 18] = [0, 0, 0, 3, 6, 7, 6, 3, 0, 0]
W1[361:420:6, 19] = [0, 0, 0, 3, 6, 7, 6, 3, 0, 0]
W1[421:480:6, 20] = [0, 0, 0, 3, 6, 7, 6, 3, 0, 0]
W1[481:540:6, 21] = [0, 0, 0, 3, 6, 7, 6, 3, 0, 0]

# Distinguish slower enemies from bullets at two-cell distance.
W1[80:105:24, 14] = -2
W1[140:165:24, 15] = -2
W1[200:225:24, 16] = -2
W1[260:285:24, 17] = -2
W1[320:345:24, 18] = -2
W1[380:405:24, 19] = -2
W1[440:465:24, 20] = -2
W1[500:525:24, 21] = -2

# Player-row indicators, using the fixed front column.
b1[22:30] = -5
W1[90, 22] = 10
W1[150, 23] = 10
W1[210, 24] = 10
W1[270, 25] = 10
W1[330, 26] = 10
W1[390, 27] = 10
W1[450, 28] = 10
W1[510, 29] = 10

# Return flag: oxygen below 60 and gauge column 8 occupied.
b1[30] = -4
W1[557, 30] = -12
W1[593, 30] = 8

# Passenger indication; ambiguous only while oxygen is nearly full.
b1[31] = -4
W1[593, 31] = 8

# Second layer: aggregate directional attractions.
W2[0:7, 0] = 0.5
W2[7:14, 1] = 0.5
b2[0:2] = 3.5

# Unsafe upward destinations: target rows 1..7.
b2[2:17] = -6
W2[14, 2] = 4
W2[23, 2] = 6
W2[15, 3] = 4
W2[24, 3] = 6
W2[16, 4] = 4
W2[25, 4] = 6
W2[17, 5] = 4
W2[26, 5] = 6
W2[18, 6] = 4
W2[27, 6] = 6
W2[19, 7] = 4
W2[28, 7] = 6
W2[20, 8] = 4
W2[29, 8] = 6

# Unsafe downward destination from the surface.
W2[14, 9] = 4
W2[22:30, 9] = -6
b2[9] = -48

# Unsafe downward destinations from rows 1..7.
W2[15, 10] = 4
W2[22, 10] = 6
W2[16, 11] = 4
W2[23, 11] = 6
W2[17, 12] = 4
W2[24, 12] = 6
W2[18, 13] = 4
W2[25, 13] = 6
W2[19, 14] = 4
W2[26, 14] = 6
W2[20, 15] = 4
W2[27, 15] = 6
W2[21, 16] = 4
W2[28, 16] = 6

# Current-row fire/hold scores: reward raises them, danger lowers them.
b2[17:25] = -11
b2[17] = -13
W2[14, 17] = -5
W2[22, 17] = 8
W2[15, 18] = -5
W2[23, 18] = 8
W2[0, 18] = 2
W2[16, 19] = -5
W2[24, 19] = 8
W2[1, 19] = 2
W2[17, 20] = -5
W2[25, 20] = 8
W2[2, 20] = 2
W2[18, 21] = -5
W2[26, 21] = 8
W2[3, 21] = 2
W2[19, 22] = -5
W2[27, 22] = 8
W2[4, 22] = 2
W2[20, 23] = -5
W2[28, 23] = 8
W2[5, 23] = 2
W2[21, 24] = -5
W2[29, 24] = 8
W2[6, 24] = 2

# Oxygen-return command.
W2[30, 25] = 4

# Leave rows 0 and 1, except while returning to the surface.
W2[23:30, 26] = -5
W2[30, 26] = -5
b2[26] = -35

# Upper guard: rows 1 and 2 always; row 3 without a passenger.
# The return flag disables this guard.
W2[22, 27] = 8
W2[23, 27] = 8
W2[24, 27] = 3
W2[31, 27] = -3
W2[30, 27] = -8
b2[27] = 5

# Bottom-boundary guard.
W2[29, 28] = 5

# Output layer.
W3[0, 2] = 48
W3[0, 4] = -48
W3[1, 2] = -48
W3[1, 4] = 48

W3[2:9, 2] = -100
W3[9:17, 4] = -100
W3[17:25, 5] = 60

W3[25, 2] = 50
W3[25, 4] = -5
W3[25, 5] = -5

W3[26, 4] = 18
W3[27, 2] = -80
W3[28, 4] = -80

# Biases compensate the inactive tanh gates.
# Horizontal movement is suppressed to preserve the firing geometry.
b3[:] = [-1000, -1000, -730, -1000, -867, 416.2]

# Sharpen action selection while retaining softmax sampling.
W3[:, :] *= 1.5
b3[:] *= 1.5
```