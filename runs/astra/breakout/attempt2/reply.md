The policy predicts the two legal paddle positions at the next bottom collision, including the walls’ repeated-edge behavior. It moves toward that interval and stops once inside it. During the lower part of an upward flight, it recenters; near the bricks, it starts anticipating the next return.

First-layer units 0–15 test the linear inequalities defining reflected landing intervals. Units 16–21 identify horizontal direction, flight phase, centering errors, and early-flight gates. The second layer combines these tests into movement decisions. Small, early-only random movements within the projected catch interval vary rebound direction without deliberately risking a late catch.

```edits
# Landing tests: vertical-coordinate terms.
W1[63:120:6, 0:16] = 20
W1[123:180:6, 0:16] = 40
W1[183:240:6, 0:16] = 60
W1[243:300:6, 0:16] = 80
W1[303:360:6, 0:16] = 100
W1[363:420:6, 0:16] = 120
W1[423:480:6, 0:16] = 140
W1[483:540:6, 0:16] = 160
W1[543:600:6, 0:16] = 180
W1[3:600:6, 0:16:8] *= -1
W1[3:600:6, 3:16:8] *= -1
W1[3:600:6, 5:16:8] *= -1
W1[3:600:6, 6:16:8] *= -1

# Landing tests: horizontal-coordinate terms.
W1[9:600:60, 0:16] += 20
W1[15:600:60, 0:16] += 40
W1[21:600:60, 0:16] += 60
W1[27:600:60, 0:16] += 80
W1[33:600:60, 0:16] += 100
W1[39:600:60, 0:16] += 120
W1[45:600:60, 0:16] += 140
W1[51:600:60, 0:16] += 160
W1[57:600:60, 0:16] += 180
W1[9:600:60, 1:16:4] -= 40
W1[9:600:60, 2:16:4] -= 40
W1[15:600:60, 1:16:4] -= 80
W1[15:600:60, 2:16:4] -= 80
W1[21:600:60, 1:16:4] -= 120
W1[21:600:60, 2:16:4] -= 120
W1[27:600:60, 1:16:4] -= 160
W1[27:600:60, 2:16:4] -= 160
W1[33:600:60, 1:16:4] -= 200
W1[33:600:60, 2:16:4] -= 200
W1[39:600:60, 1:16:4] -= 240
W1[39:600:60, 2:16:4] -= 240
W1[45:600:60, 1:16:4] -= 280
W1[45:600:60, 2:16:4] -= 280
W1[51:600:60, 1:16:4] -= 320
W1[51:600:60, 2:16:4] -= 320
W1[57:600:60, 1:16:4] -= 360
W1[57:600:60, 2:16:4] -= 360

# Landing tests: paddle-coordinate terms.
W1[546, 0:16] = -20
W1[552, 0:16] = -40
W1[558, 0:16] = -60
W1[564, 0:16] = -80
W1[570, 0:16] = -100
W1[576, 0:16] = -120
W1[582, 0:16] = -140
W1[588, 0:16] = -160
W1[594, 0:16] = -180
W1[540:600:6, 2:16:4] *= -1
W1[540:600:6, 3:16:4] *= -1
b1[0:16] = [150, 190, -190, -230, -190, 130, 150, -170, 170, 210, -170, -210, -170, 150, 170, -150]

# Horizontal direction, including zero-displacement wall reflections.
W1[3:600:60, 16] = 10
W1[9:600:60, 16] = 20
W1[15:600:60, 16] = 40
W1[21:600:60, 16] = 60
W1[27:600:60, 16] = 80
W1[33:600:60, 16] = 100
W1[39:600:60, 16] = 120
W1[45:600:60, 16] = 140
W1[51:600:60, 16] = 160
W1[57:600:60, 16] = 170
W1[10:600:60, 16] = -20
W1[16:600:60, 16] = -40
W1[22:600:60, 16] = -60
W1[28:600:60, 16] = -80
W1[34:600:60, 16] = -100
W1[40:600:60, 16] = -120
W1[46:600:60, 16] = -140
W1[52:600:60, 16] = -160
W1[58:600:60, 16] = -180

# Forecast phase: descending or upper field; otherwise recenter.
W1[3:60:6, 17] = 30
W1[63:120:6, 17] = 50
W1[123:180:6, 17] = 70
W1[183:240:6, 17] = 90
W1[243:360:6, 17] = 110
W1[363:420:6, 17] = 130
W1[423:540:6, 17] = 150
W1[543:600:6, 17] = 170
W1[64:120:6, 17] = -20
W1[124:180:6, 17] = -40
W1[184:240:6, 17] = -60
W1[244:300:6, 17] = -80
W1[304:360:6, 17] = -100
W1[364:420:6, 17] = -120
W1[424:480:6, 17] = -140
W1[484:540:6, 17] = -160
W1[544:600:6, 17] = -180

# Earliest-flight gate: rows 0–2, or upward travel at row 3.
W1[3:60:6, 18] = 30
W1[63:120:6, 18] = 10
W1[123:180:6, 18] = -10
W1[183:240:6, 18] = -70
W1[243:300:6, 18] = -110
W1[303:360:6, 18] = -130
W1[363:420:6, 18] = -150
W1[423:480:6, 18] = -170
W1[483:540:6, 18] = -190
W1[543:600:6, 18] = -210
W1[64:120:6, 18] = 20
W1[124:180:6, 18] = 40
W1[184:240:6, 18] = 60
W1[244:300:6, 18] = 80
W1[304:360:6, 18] = 100
W1[364:420:6, 18] = 120
W1[424:480:6, 18] = 140
W1[484:540:6, 18] = 160
W1[544:600:6, 18] = 180

# Centering tests and upper-field gate.
W1[540:600:6, 19] = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]
W1[540:600:6, 20] = [0, -20, -40, -60, -80, -100, -120, -140, -160, -180]
b1[19:22] = [-110, 70, 90]
W1[63:120:6, 21] = -20
W1[123:180:6, 21] = -40
W1[183:240:6, 21] = -60
W1[243:300:6, 21] = -80
W1[303:360:6, 21] = -100
W1[363:420:6, 21] = -120
W1[423:480:6, 21] = -140
W1[483:540:6, 21] = -160
W1[543:600:6, 21] = -180

# Mandatory right movement for rightward flight.
W2[0:2, 0] = 10
W2[16:18, 0] = 10

# Mandatory left movement for rightward flight.
W2[2:4, 1] = 10
W2[16:18, 1] = 20

# Mandatory left movement for leftward flight.
W2[6:8, 2] = 10
W2[16, 2] = -10
W2[17, 2] = 10

# Mandatory right movement for leftward flight.
W2[4:6, 3] = 10
W2[16, 3] = -20
W2[17, 3] = 20

# Centering movements.
W2[19, 4] = 10
W2[17, 4] = -10
W2[20, 5] = 10
W2[17, 5] = -10

# Early optional left movement for rightward flight.
W2[10:12, 6] = 10
W2[16:18, 6] = 20
W2[21, 6] = 20

# Early optional left movement for leftward flight.
W2[14:16, 7] = 10
W2[16, 7] = -10
W2[17, 7] = 10
W2[21, 7] = 10

# Early optional right movement for rightward flight.
W2[8:10, 8] = 10
W2[16:18, 8] = 10
W2[21, 8] = 10

# Early optional right movement for leftward flight.
W2[12:14, 9] = 10
W2[16, 9] = -20
W2[17, 9] = 20
W2[21, 9] = 20

# Extra exploration only very early in flight.
W2[18, 10] = 10
b2[0:10] = [-30, -30, -30, -30, -10, -10, -50, -40, -40, -50]

# Mandatory decisions dominate; optional moves only compete with no-op.
W3[0, 3] = 24
W3[1:3, 1] = 24
W3[3, 3] = 24
W3[4, 1] = 24
W3[5, 3] = 24
W3[6:8, 1] = 10
W3[8:10, 3] = 10
W3[10, 1] = 2
W3[10, 3] = 2
b3[0:6] = [0, 70, -30, 70, -30, -30]
```