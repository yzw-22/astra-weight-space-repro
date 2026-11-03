A simple hand-constructed policy:
- Compute signed row/column differences from the player to treasure and enemies using weighted sums over the grid.
- Gate those differences by “treasure exists”/“enemy exists” units in the second hidden layer, producing four directional attraction units (toward treasure) and four repulsion units (away from enemies).
- Detect whether an enemy shares the player’s row, and if so strongly encourage vertical escape.
- Add boundary penalties so the policy does not select an action that would only push into a wall.
- Use no-op when there is neither treasure nor enemy.

```edits
# ---- W1: first-layer feature detectors ----

# Player row one-hot units 0..9 (0 when row not active, +1 when active)
W1[0:60:6, 0] = 4
W1[60:120:6, 1] = 4
W1[120:180:6, 2] = 4
W1[180:240:6, 3] = 4
W1[240:300:6, 4] = 4
W1[300:360:6, 5] = 4
W1[360:420:6, 6] = 4
W1[420:480:6, 7] = 4
W1[480:540:6, 8] = 4
W1[540:600:6, 9] = 4

# Enemy row detectors for rows 1..8 into units 10..17
W1[61:120:6, 10] = 4
W1[121:180:6, 11] = 4
W1[181:240:6, 12] = 4
W1[241:300:6, 13] = 4
W1[301:360:6, 14] = 4
W1[361:420:6, 15] = 4
W1[421:480:6, 16] = 4
W1[481:540:6, 17] = 4

# Treasure exists / enemy exists
W1[2:600:6, 18] = 4
W1[1:600:6, 21] = 4
b1[18] = -2
b1[21] = -2

# Treasure vertical difference: +0.5*row for treasure, -0.5*row for player
W1[62:120:6, 19] = 0.5
W1[60:120:6, 19] = -0.5
W1[122:180:6, 19] = 1.0
W1[120:180:6, 19] = -1.0
W1[182:240:6, 19] = 1.5
W1[180:240:6, 19] = -1.5
W1[242:300:6, 19] = 2.0
W1[240:300:6, 19] = -2.0
W1[302:360:6, 19] = 2.5
W1[300:360:6, 19] = -2.5
W1[362:420:6, 19] = 3.0
W1[360:420:6, 19] = -3.0
W1[422:480:6, 19] = 3.5
W1[420:480:6, 19] = -3.5
W1[482:540:6, 19] = 4.0
W1[480:540:6, 19] = -4.0

# Treasure horizontal difference: +0.5*col for treasure, -0.5*col for player
W1[8:600:60, 20] = 0.5
W1[6:600:60, 20] = -0.5
W1[14:600:60, 20] = 1.0
W1[12:600:60, 20] = -1.0
W1[20:600:60, 20] = 1.5
W1[18:600:60, 20] = -1.5
W1[26:600:60, 20] = 2.0
W1[24:600:60, 20] = -2.0
W1[32:600:60, 20] = 2.5
W1[30:600:60, 20] = -2.5
W1[38:600:60, 20] = 3.0
W1[36:600:60, 20] = -3.0
W1[44:600:60, 20] = 3.5
W1[42:600:60, 20] = -3.5
W1[50:600:60, 20] = 4.0
W1[48:600:60, 20] = -4.0
W1[56:600:60, 20] = 4.5
W1[54:600:60, 20] = -4.5

# Enemy vertical difference: +0.5*row for enemy, -0.5*row for player
W1[61:120:6, 22] = 0.5
W1[60:120:6, 22] = -0.5
W1[121:180:6, 22] = 1.0
W1[120:180:6, 22] = -1.0
W1[181:240:6, 22] = 1.5
W1[180:240:6, 22] = -1.5
W1[241:300:6, 22] = 2.0
W1[240:300:6, 22] = -2.0
W1[301:360:6, 22] = 2.5
W1[300:360:6, 22] = -2.5
W1[361:420:6, 22] = 3.0
W1[360:420:6, 22] = -3.0
W1[421:480:6, 22] = 3.5
W1[420:480:6, 22] = -3.5
W1[481:540:6, 22] = 4.0
W1[480:540:6, 22] = -4.0

# Enemy horizontal difference: +0.5*col for enemy, -0.5*col for player
W1[7:600:60, 23] = 0.5
W1[6:600:60, 23] = -0.5
W1[13:600:60, 23] = 1.0
W1[12:600:60, 23] = -1.0
W1[19:600:60, 23] = 1.5
W1[18:600:60, 23] = -1.5
W1[25:600:60, 23] = 2.0
W1[24:600:60, 23] = -2.0
W1[31:600:60, 23] = 2.5
W1[30:600:60, 23] = -2.5
W1[37:600:60, 23] = 3.0
W1[36:600:60, 23] = -3.0
W1[43:600:60, 23] = 3.5
W1[42:600:60, 23] = -3.5
W1[49:600:60, 23] = 4.0
W1[48:600:60, 23] = -4.0
W1[55:600:60, 23] = 4.5
W1[54:600:60, 23] = -4.5

# Boundary detectors
W1[60:120:6, 24] = 4
W1[480:540:6, 25] = 4
W1[0:600:60, 26] = 4
W1[54:600:60, 27] = 4

# ---- W2: gated directional units and same-row detectors ----

# Gated enemy repulsion units 0..3
W2[21, 0] = 2
W2[22, 0] = 2
W2[21, 1] = 2
W2[22, 1] = -2
W2[21, 2] = 2
W2[23, 2] = 2
W2[21, 3] = 2
W2[23, 3] = -2
b2[0:8] = -2

# Gated treasure attraction units 4..7
W2[18, 4] = 2
W2[19, 4] = 2
W2[18, 5] = 2
W2[19, 5] = -2
W2[18, 6] = 2
W2[20, 6] = 2
W2[18, 7] = 2
W2[20, 7] = -2

# Pass-through units 8..13
W2[18, 8] = 1.5
W2[21, 9] = 1.5
W2[24, 10] = 1.5
W2[25, 11] = 1.5
W2[26, 12] = 1.5
W2[27, 13] = 1.5

# Same-row detectors for rows 1..8 into h2 units 14..21
W2[1, 14] = 2
W2[10, 14] = 2
W2[2, 15] = 2
W2[11, 15] = 2
W2[3, 16] = 2
W2[12, 16] = 2
W2[4, 17] = 2
W2[13, 17] = 2
W2[5, 18] = 2
W2[14, 18] = 2
W2[6, 19] = 2
W2[15, 19] = 2
W2[7, 20] = 2
W2[16, 20] = 2
W2[8, 21] = 2
W2[17, 21] = 2
b2[14:22] = -3

# ---- W3: output logits ----

# No-op: high when neither treasure nor enemy exists
W3[8, 0] = -5
W3[9, 0] = -5

# Left: treasure left + enemy right repulsion - left wall
W3[7, 1] = 4
W3[2, 1] = 5
W3[12, 1] = -6

# Up: treasure up + enemy below repulsion + same-row escape - top wall
W3[5, 2] = 4
W3[0, 2] = 5
W3[10, 2] = -6
W3[14:22, 2] = 5

# Right: treasure right + enemy left repulsion - right wall
W3[6, 3] = 4
W3[3, 3] = 5
W3[13, 3] = -6

# Down: treasure down + enemy above repulsion + same-row escape - bottom wall
W3[4, 4] = 4
W3[1, 4] = 5
W3[11, 4] = -6
W3[14:22, 4] = 5

# Fire = no-op but slightly worse
W3[8, 5] = -5
W3[9, 5] = -5
b3[5] = -1
```