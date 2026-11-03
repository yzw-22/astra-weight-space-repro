The policy stabilizes the submarine at column 2, facing right, and fires while waiting. It targets the uppermost enemy/diver in rows 3–8, with extra priority for nearby divers in the lower rows. It avoids immediate collision cells and approaching bullets. Keeping a two-row surface buffer reduces accidental surfacing from sticky actions; low oxygen plus an observable carried diver triggers ascent.

The first layer computes gated target directions, row-wise danger signals, horizontal-position corrections, gauge conditions, and small encodings of player height and squared height. The second layer combines these into target commands and row-specific collision vetoes. Collision avoidance outranks targeting; oxygen ascent outranks diver pursuit. These are hand-designed heuristics, without an empirical optimality claim.

```edits
# First-layer biases
b1[0:15] = -133
b1[15:23] = -4
b1[25:29] = -4
b1[29:31] = -133
b1[31] = -8

# Primary upward targets: uppermost reward-bearing row among rows 3..8
W1[182:240:6, 0] = 10
W1[240:540:6, 0] = 64

W1[182:242:6, 1] = -128
W1[242:300:6, 1] = 10
W1[300:540:6, 1] = 64

W1[182:302:6, 2] = -128
W1[302:360:6, 2] = 10
W1[360:540:6, 2] = 64

W1[182:362:6, 3] = -128
W1[362:420:6, 3] = 10
W1[420:540:6, 3] = 64

W1[182:422:6, 4] = -128
W1[422:480:6, 4] = 10
W1[480:540:6, 4] = 64

# Primary downward targets
W1[182:240:6, 5] = 10
W1[0:180:6, 5] = 64

W1[182:242:6, 6] = -128
W1[242:300:6, 6] = 10
W1[0:240:6, 6] = 64

W1[182:302:6, 7] = -128
W1[302:360:6, 7] = 10
W1[0:300:6, 7] = 64

W1[182:362:6, 8] = -128
W1[362:420:6, 8] = 10
W1[0:360:6, 8] = 64

W1[182:422:6, 9] = -128
W1[422:480:6, 9] = 10
W1[0:420:6, 9] = 64

W1[182:482:6, 10] = -128
W1[482:540:6, 10] = 10
W1[0:480:6, 10] = 64

# Nearby lower-row divers: upward-or-aligned signals
W1[362:392:6, 11] = 10
W1[361:391:6, 11] = -10
W1[360:540:6, 11] = 64

W1[422:452:6, 12] = 10
W1[421:451:6, 12] = -10
W1[420:540:6, 12] = 64

W1[482:512:6, 13] = 10
W1[481:511:6, 13] = -10
W1[480:540:6, 13] = 64

# Nearby lower-row divers: downward-or-aligned signals
W1[362:392:6, 14] = 10
W1[361:391:6, 14] = -10
W1[0:420:6, 14] = 64

W1[422:452:6, 29] = 10
W1[421:451:6, 29] = -10
W1[0:480:6, 29] = 64

W1[482:512:6, 30] = 10
W1[481:511:6, 30] = -10
W1[0:540:6, 30] = 64

# Danger in columns 1..3; additional bullet warning in columns 0 and 4
W1[61:91:6, 15] = 12
W1[61:86:24, 15] = 6
W1[62:87:24, 15] = -6

W1[121:151:6, 16] = 12
W1[121:146:24, 16] = 6
W1[122:147:24, 16] = -6

W1[181:211:6, 17] = 12
W1[181:206:24, 17] = 6
W1[182:207:24, 17] = -6

W1[241:271:6, 18] = 12
W1[241:266:24, 18] = 6
W1[242:267:24, 18] = -6

W1[301:331:6, 19] = 12
W1[301:326:24, 19] = 6
W1[302:327:24, 19] = -6

W1[361:391:6, 20] = 12
W1[361:386:24, 20] = 6
W1[362:387:24, 20] = -6

W1[421:451:6, 21] = 12
W1[421:446:24, 21] = 6
W1[422:447:24, 21] = -6

W1[481:511:6, 22] = 12
W1[481:506:24, 22] = 6
W1[482:507:24, 22] = -6

# Small player-height and squared-height encodings
W1[0:60:6, 23] = -0.002
W1[60:120:6, 23] = -0.0015
W1[120:180:6, 23] = -0.001
W1[180:240:6, 23] = -0.0005
W1[300:360:6, 23] = 0.0005
W1[360:420:6, 23] = 0.001
W1[420:480:6, 23] = 0.0015
W1[480:540:6, 23] = 0.002

W1[0:60:6, 24] = 0.008
W1[60:120:6, 24] = 0.0045
W1[120:180:6, 24] = 0.002
W1[180:240:6, 24] = 0.0005
W1[300:360:6, 24] = 0.0005
W1[360:420:6, 24] = 0.002
W1[420:480:6, 24] = 0.0045
W1[480:540:6, 24] = 0.008

# Oxygen below 60 and at least one visible carried-diver gauge cell
W1[557, 25] = -16
W1[593, 25] = 8

# Horizontal stabilization: move left when any player cell is beyond column 2
W1[18:600:60, 26] = 8
W1[24:600:60, 26] = 8
W1[30:600:60, 26] = 8
W1[36:600:60, 26] = 8
W1[42:600:60, 26] = 8
W1[48:600:60, 26] = 8
W1[54:600:60, 26] = 8

# Move right from column 0, or turn right from the left-facing column-1 configuration
W1[0:600:60, 27] = 8
W1[6:600:60, 28] = 8
W1[10:600:60, 28] = -16

# Recover downward from shallow rows after unintended sticky upward moves
W1[60:180:6, 31] = 8

# Second layer: danger in the destination of an upward move
W2[23, 0:7] = [-48000, -24000, 0, 24000, 48000, 72000, 96000]
W2[24, 0:23] = -12000
b2[0:7] = [-51, -15, -3, -15, -51, -111, -195]
W2[15, 0] = 6
W2[16, 1] = 6
W2[17, 2] = 6
W2[18, 3] = 6
W2[19, 4] = 6
W2[20, 5] = 6
W2[21, 6] = 6

# Danger in the destination of a downward move
W2[23, 7:15] = [-96000, -72000, -48000, -24000, 0, 24000, 48000, 72000]
b2[7:15] = [-195, -111, -51, -15, -3, -15, -51, -111]
W2[15, 7] = 6
W2[16, 8] = 6
W2[17, 9] = 6
W2[18, 10] = 6
W2[19, 11] = 6
W2[20, 12] = 6
W2[21, 13] = 6
W2[22, 14] = 6

# Danger in the player's current row
W2[23, 15:23] = [-72000, -48000, -24000, 0, 24000, 48000, 72000, 96000]
b2[15:23] = [-111, -51, -15, -3, -15, -51, -111, -195]
W2[15, 15] = 6
W2[16, 16] = 6
W2[17, 17] = 6
W2[18, 18] = 6
W2[19, 19] = 6
W2[20, 20] = 6
W2[21, 21] = 6
W2[22, 22] = 6

# Aggregate primary target directions
W2[0:5, 23] = 4
b2[23] = 16
W2[5:11, 24] = 4
W2[31, 24] = 4
b2[24] = 24

# Oxygen ascent
W2[25, 25] = 4

# Forbid upward moves from rows 1..3 unless oxygen ascent is enabled
W2[25, 26] = -12
W2[23, 26] = -48000
W2[24, 26] = -12000
b2[26] = -45

# Horizontal correction commands
W2[26, 27] = 4
W2[27:29, 28] = 4
b2[28] = 4

# Aggregate urgent diver directions; both activate when aligned
W2[11:14, 29] = 4
b2[29] = 8
W2[14, 30] = 4
W2[29:31, 30] = 4
b2[30] = 8

# Forbid a downward move that would remain clamped at row 8
W2[23, 31] = 96000
W2[24, 31] = -12000
b2[31] = -189

# Output biases cancel the inactive detector baselines
b3[:] = [-150, 0, -1380, 0, -1660, -464]

# Destination collision vetoes
W3[0:7, 2] = -200
W3[7:15, 4] = -200

# Leave a dangerous current row rather than continuing to fire
W3[15:23, 2] = 20
W3[15:23, 4] = 20
W3[15:23, 5] = -60

# Primary pursuit
W3[23, 2] = 10
W3[24, 4] = 10

# Oxygen ascent and surface buffer
W3[25, 2] = 80
W3[26, 2] = -200

# Horizontal stabilization
W3[27, 1] = 120
W3[28, 3] = 120

# Urgent diver pursuit; alignment favors firing in place while awaiting pickup
W3[29, 2] = 20
W3[29, 4] = -40
W3[29, 5] = 6
W3[30, 2] = -40
W3[30, 4] = 20
W3[30, 5] = 6

# Bottom boundary
W3[31, 4] = -200
```