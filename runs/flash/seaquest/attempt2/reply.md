Design: I encode the submarine’s front/back position and facing from the player and trail channels, detect targets below, in the current row, and above with row-gated features, and steer toward targets while suppressing descent when an enemy/diver is already in the current row. Firing is enabled only when a current-row target and a danger cell are present, with separate right/left aim units using the submarine’s facing. Oxygen-danger surfacing overrides all other behavior when oxygen is low and the diver gauge shows at least one carried diver. A shield against trail-contaminated orientation suppresses firing when the front/back column detection is inconsistent.

```edits
# W1: player back-column detectors (B_c), c=0..9
W1[0:600:60, 0] = 10
W1[4:600:60, 0] = 5
b1[0] = -12.5
W1[6:600:60, 1] = 10
W1[10:600:60, 1] = 5
b1[1] = -12.5
W1[12:600:60, 2] = 10
W1[16:600:60, 2] = 5
b1[2] = -12.5
W1[18:600:60, 3] = 10
W1[22:600:60, 3] = 5
b1[3] = -12.5
W1[24:600:60, 4] = 10
W1[28:600:60, 4] = 5
b1[4] = -12.5
W1[30:600:60, 5] = 10
W1[34:600:60, 5] = 5
b1[5] = -12.5
W1[36:600:60, 6] = 10
W1[40:600:60, 6] = 5
b1[6] = -12.5
W1[42:600:60, 7] = 10
W1[46:600:60, 7] = 5
b1[7] = -12.5
W1[48:600:60, 8] = 10
W1[52:600:60, 8] = 5
b1[8] = -12.5
W1[54:600:60, 9] = 10
W1[58:600:60, 9] = 5
b1[9] = -12.5

# PX = small linear encoding of column-sum of player channel
W1[6:600:60, 10] = 0.02
W1[12:600:60, 10] = 0.04
W1[18:600:60, 10] = 0.06
W1[24:600:60, 10] = 0.08
W1[30:600:60, 10] = 0.10
W1[36:600:60, 10] = 0.12
W1[42:600:60, 10] = 0.14
W1[48:600:60, 10] = 0.16
W1[54:600:60, 10] = 0.18

# TX = small linear encoding of column-sum of reward/target channel
W1[8:600:60, 11] = 0.02
W1[14:600:60, 11] = 0.04
W1[20:600:60, 11] = 0.06
W1[26:600:60, 11] = 0.08
W1[32:600:60, 11] = 0.10
W1[38:600:60, 11] = 0.12
W1[44:600:60, 11] = 0.14
W1[50:600:60, 11] = 0.16
W1[56:600:60, 11] = 0.18

# Down_r: target in row r, player strictly above row r
W1[0:60:6, 12] = 50
W1[62:120:6, 12] = 5
b1[12] = -102.5
W1[0:120:6, 13] = 50
W1[122:180:6, 13] = 5
b1[13] = -102.5
W1[0:180:6, 14] = 50
W1[182:240:6, 14] = 5
b1[14] = -102.5
W1[0:240:6, 15] = 50
W1[242:300:6, 15] = 5
b1[15] = -102.5
W1[0:300:6, 16] = 50
W1[302:360:6, 16] = 5
b1[16] = -102.5
W1[0:360:6, 17] = 50
W1[362:420:6, 17] = 5
b1[17] = -102.5
W1[0:420:6, 18] = 50
W1[422:480:6, 18] = 5
b1[18] = -102.5
W1[0:480:6, 19] = 50
W1[482:540:6, 19] = 5
b1[19] = -102.5

# Current_r: player and a target in the same row r
W1[60:120:6, 20] = 50
W1[62:120:6, 20] = 5
b1[20] = -102.5
W1[120:180:6, 21] = 50
W1[122:180:6, 21] = 5
b1[21] = -102.5
W1[180:240:6, 22] = 50
W1[182:240:6, 22] = 5
b1[22] = -102.5
W1[240:300:6, 23] = 50
W1[242:300:6, 23] = 5
b1[23] = -102.5
W1[300:360:6, 24] = 50
W1[302:360:6, 24] = 5
b1[24] = -102.5
W1[360:420:6, 25] = 50
W1[362:420:6, 25] = 5
b1[25] = -102.5
W1[420:480:6, 26] = 50
W1[422:480:6, 26] = 5
b1[26] = -102.5
W1[480:540:6, 27] = 50
W1[482:540:6, 27] = 5
b1[27] = -102.5

# Any target present
W1[2:600:6, 28] = 5
b1[28] = -2.5

# Oxygen below 60: gauge column 2 absent
W1[557, 29] = -10
b1[29] = 5

# Any carried diver shown in low-oxygen gauge region (cols 3..8)
W1[563:594:6, 30] = 5
b1[30] = -2.5

# Any danger present
W1[1:600:6, 31] = 5
b1[31] = -2.5

# W2: second-layer conditions
W2[0, 12:20] = 5
b2[0] = 37.5

W2[1, 28] = 5
W2[1, 12:20] = -2.5
W2[1, 20:28] = -2.5
b2[1] = -42.5

W2[2, 29] = 5
W2[2, 30] = 5
b2[2] = -7.5

W2[3, 20:28] = 5
b2[3] = 37.5

W2[4, 10] = 250
W2[4, 0:10] = [0, -5, -10, -15, -20, -25, -30, -35, -40, -45]
b2[4] = -225

W2[5, 28] = 5

W2[6, 28] = 20
W2[6, 11] = 500
W2[6, 10] = -250
b2[6] = -26

W2[7, 28] = 20
W2[7, 11] = -500
W2[7, 10] = 250
b2[7] = -26

W2[8, 10] = 4500
W2[8, 0:10] = [0, -95, -190, -285, -380, -475, -570, -665, -760, -855]
W2[8, 11] = 500
W2[8, 20:28] = 20
W2[8, 31] = 20
b2[8] = -4270

W2[9, 10] = -4500
W2[9, 0:10] = [0, 95, 190, 285, 380, 475, 570, 665, 760, 855]
W2[9, 11] = -500
W2[9, 20:28] = 20
W2[9, 31] = 20
b2[9] = 4280

W2[10, 20:28] = 20
W2[10, 31] = 20
b2[10] = 130

W2[11, 0:10] = 5
b2[11] = 35

# W3 and b3
W3[7, 1] = 4.0
W3[2, 2] = 30.0
W3[1, 2] = 5.0
W3[6, 3] = 4.0
W3[0, 4] = 6.0
W3[3, 4] = -5.5
W3[8, 5] = 10.0
W3[9, 5] = 10.0
W3[10, 5] = 1.5
W3[11, 5] = -15.0

b3[0] = 2.0
b3[1] = 4.0
b3[2] = 35.0
b3[3] = 4.0
b3[4] = 0.5
b3[5] = 6.5
```