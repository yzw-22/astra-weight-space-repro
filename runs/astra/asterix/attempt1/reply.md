I use a **fixed-column interceptor**: stay at column 5, move vertically toward incoming treasure, and wait when aligned with it. Trails distinguish incoming treasure from treasure that has already passed the interception column.

The first layer computes row-specific hazards for up, down, and staying, plus treasure-gated vertical displacement. Hazard features include approaching enemies and the extra vertical step a sticky action could cause. The second layer separates immediate danger from precautionary warnings and decodes treasure as above, below, or aligned. Safety penalties dominate reward-seeking; a small upward preference breaks target ties.

```edits
# First-layer hazard units: up 0:8, down 8:16, stay 16:24.
b1[0:24] = -36

# Row 1 enemy and trail contributions.
W1[79:109:6, 0:24:8] = [[4, 4, 4], [8, 8, 8], [24, 24, 24], [8, 8, 8], [4, 4, 4]]
W1[76:118:6, 0:24:8] = [[8, 8, 8], [8, 8, 8], [-8, -8, -8], [-8, -8, -8], [-8, -8, -8], [8, 8, 8], [8, 8, 8]]

# Row 2.
W1[139:169:6, 1:24:8] = [[4, 4, 4], [8, 8, 8], [24, 24, 24], [8, 8, 8], [4, 4, 4]]
W1[136:178:6, 1:24:8] = [[8, 8, 8], [8, 8, 8], [-8, -8, -8], [-8, -8, -8], [-8, -8, -8], [8, 8, 8], [8, 8, 8]]

# Row 3.
W1[199:229:6, 2:24:8] = [[4, 4, 4], [8, 8, 8], [24, 24, 24], [8, 8, 8], [4, 4, 4]]
W1[196:238:6, 2:24:8] = [[8, 8, 8], [8, 8, 8], [-8, -8, -8], [-8, -8, -8], [-8, -8, -8], [8, 8, 8], [8, 8, 8]]

# Row 4.
W1[259:289:6, 3:24:8] = [[4, 4, 4], [8, 8, 8], [24, 24, 24], [8, 8, 8], [4, 4, 4]]
W1[256:298:6, 3:24:8] = [[8, 8, 8], [8, 8, 8], [-8, -8, -8], [-8, -8, -8], [-8, -8, -8], [8, 8, 8], [8, 8, 8]]

# Row 5.
W1[319:349:6, 4:24:8] = [[4, 4, 4], [8, 8, 8], [24, 24, 24], [8, 8, 8], [4, 4, 4]]
W1[316:358:6, 4:24:8] = [[8, 8, 8], [8, 8, 8], [-8, -8, -8], [-8, -8, -8], [-8, -8, -8], [8, 8, 8], [8, 8, 8]]

# Row 6.
W1[379:409:6, 5:24:8] = [[4, 4, 4], [8, 8, 8], [24, 24, 24], [8, 8, 8], [4, 4, 4]]
W1[376:418:6, 5:24:8] = [[8, 8, 8], [8, 8, 8], [-8, -8, -8], [-8, -8, -8], [-8, -8, -8], [8, 8, 8], [8, 8, 8]]

# Row 7.
W1[439:469:6, 6:24:8] = [[4, 4, 4], [8, 8, 8], [24, 24, 24], [8, 8, 8], [4, 4, 4]]
W1[436:478:6, 6:24:8] = [[8, 8, 8], [8, 8, 8], [-8, -8, -8], [-8, -8, -8], [-8, -8, -8], [8, 8, 8], [8, 8, 8]]

# Row 8.
W1[499:529:6, 7:24:8] = [[4, 4, 4], [8, 8, 8], [24, 24, 24], [8, 8, 8], [4, 4, 4]]
W1[496:538:6, 7:24:8] = [[8, 8, 8], [8, 8, 8], [-8, -8, -8], [-8, -8, -8], [-8, -8, -8], [8, 8, 8], [8, 8, 8]]

# Up destinations and possible extra sticky-action destinations.
# At the top boundary, up remains in row 1.
W1[60:180:6, 0] = 24
W1[180:240:6, 0] = 22
W1[180:240:6, 1] = 24
W1[240:300:6, 1] = 22
W1[240:300:6, 2] = 24
W1[300:360:6, 2] = 22
W1[300:360:6, 3] = 24
W1[360:420:6, 3] = 22
W1[360:420:6, 4] = 24
W1[420:480:6, 4] = 22
W1[420:480:6, 5] = 24
W1[480:540:6, 5] = 22
W1[480:540:6, 6] = 24

# Down destinations and possible extra sticky-action destinations.
# At the bottom boundary, down remains in row 8.
W1[60:120:6, 9] = 24
W1[120:180:6, 10] = 24
W1[60:120:6, 10] = 22
W1[180:240:6, 11] = 24
W1[120:180:6, 11] = 22
W1[240:300:6, 12] = 24
W1[180:240:6, 12] = 22
W1[300:360:6, 13] = 24
W1[240:300:6, 13] = 22
W1[360:420:6, 14] = 24
W1[300:360:6, 14] = 22
W1[420:540:6, 15] = 24
W1[360:420:6, 15] = 22

# Staying: select the player's current row.
W1[60:120:6, 16] = 24
W1[120:180:6, 17] = 24
W1[180:240:6, 18] = 24
W1[240:300:6, 19] = 24
W1[300:360:6, 20] = 24
W1[360:420:6, 21] = 24
W1[420:480:6, 22] = 24
W1[480:540:6, 23] = 24

# Treasure units 24:32 encode vertical displacement when treasure is incoming.
b1[24:32] = [-8.15, -8.30, -8.45, -8.60, -8.75, -8.90, -9.05, -9.20]

# Player vertical coordinate, shared by all treasure units.
W1[60:120:6, 24:32] = 0.15
W1[120:180:6, 24:32] = 0.30
W1[180:240:6, 24:32] = 0.45
W1[240:300:6, 24:32] = 0.60
W1[300:360:6, 24:32] = 0.75
W1[360:420:6, 24:32] = 0.90
W1[420:480:6, 24:32] = 1.05
W1[480:540:6, 24:32] = 1.20
W1[540:600:6, 24:32] = 1.35

# Treasure and trail weights reject treasure moving away from column 5.
W1[62:122:6, 24] = [8, 8, 16, 16, 24, 24, 24, 16, 16, 8]
W1[64:124:6, 24] = [0, -8, -8, -16, -16, -24, -16, -16, -8, -8]
W1[122:182:6, 25] = [8, 8, 16, 16, 24, 24, 24, 16, 16, 8]
W1[124:184:6, 25] = [0, -8, -8, -16, -16, -24, -16, -16, -8, -8]
W1[182:242:6, 26] = [8, 8, 16, 16, 24, 24, 24, 16, 16, 8]
W1[184:244:6, 26] = [0, -8, -8, -16, -16, -24, -16, -16, -8, -8]
W1[242:302:6, 27] = [8, 8, 16, 16, 24, 24, 24, 16, 16, 8]
W1[244:304:6, 27] = [0, -8, -8, -16, -16, -24, -16, -16, -8, -8]
W1[302:362:6, 28] = [8, 8, 16, 16, 24, 24, 24, 16, 16, 8]
W1[304:364:6, 28] = [0, -8, -8, -16, -16, -24, -16, -16, -8, -8]
W1[362:422:6, 29] = [8, 8, 16, 16, 24, 24, 24, 16, 16, 8]
W1[364:424:6, 29] = [0, -8, -8, -16, -16, -24, -16, -16, -8, -8]
W1[422:482:6, 30] = [8, 8, 16, 16, 24, 24, 24, 16, 16, 8]
W1[424:484:6, 30] = [0, -8, -8, -16, -16, -24, -16, -16, -8, -8]
W1[482:542:6, 31] = [8, 8, 16, 16, 24, 24, 24, 16, 16, 8]
W1[484:544:6, 31] = [0, -8, -8, -16, -16, -24, -16, -16, -8, -8]

# Second-layer treasure decoding: three thresholds per row.
W2[24, 0:3] = [40, 40, 50]
W2[25, 3:6] = [40, 40, 50]
W2[26, 6:9] = [40, 40, 50]
W2[27, 9:12] = [40, 40, 50]
W2[28, 12:15] = [40, 40, 50]
W2[29, 15:18] = [40, 40, 50]
W2[30, 18:21] = [40, 40, 50]
W2[31, 21:24] = [40, 40, 50]
b2[0:24:3] = -3
b2[1:24:3] = 3
b2[2:24:3] = 45

# Aggregate precautionary and immediate hazards for each action.
W2[0:8, 24] = 200
W2[0:8, 25] = 250
W2[8:16, 26] = 200
W2[8:16, 27] = 250
W2[16:24, 28] = 200
W2[16:24, 29] = 250
b2[24:30] = [1597, 1505, 1597, 1505, 1597, 1505]

# Detect whether any incoming treasure is available.
W2[24:32, 30] = 20
b2[30] = 158

# Treasure above encourages up; treasure below encourages down.
W3[0:24:3, 2] = 6
W3[1:24:3, 4] = -6
W3[2:24:3, 4] = 6

# Strong waiting bonus when vertically aligned with incoming treasure.
W3[0:24:3, 0] = -18
W3[1:24:3, 0] = 18

# Safety penalties: immediate hazards dominate all treasure incentives.
W3[24, 2] = -40
W3[25, 2] = -400
W3[26, 4] = -40
W3[27, 4] = -400
W3[28, 0] = -70
W3[29, 0] = -400

# Prefer quiet waiting when no incoming treasure exists.
W3[30, 0] = -3

# Offsets, slight upward tie-break, and disabled horizontal/fire actions.
b3[:] = [-467, -10000, -395, -10000, -446, -10000]
```