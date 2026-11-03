The policy tracks the ball’s column and the paddle’s column. Two first-layer units compute soft column features by weighting the active paddle cell and the active ball cell by their column index. A second-layer unit compares these features, saturating to +1 when the paddle is right of the ball and -1 when it is left. The outputs then strongly prefer left or right to move the paddle under the ball, while the no-op bias dominates when the two columns are equal.

```edits
# Paddle x-coordinate feature: channel 0, row 9, columns 0..9
W1[540:600:6, 0] = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]

# Ball x-coordinate feature: channel 3, all rows, by column
W1[9:600:60, 1] = 0.05
W1[15:600:60, 1] = 0.10
W1[21:600:60, 1] = 0.15
W1[27:600:60, 1] = 0.20
W1[33:600:60, 1] = 0.25
W1[39:600:60, 1] = 0.30
W1[45:600:60, 1] = 0.35
W1[51:600:60, 1] = 0.40
W1[57:600:60, 1] = 0.45

# Compare paddle_x - ball_x in the second layer
W2[0:2, 0] = [40.0, -40.0]

# Outputs: left if paddle is right of ball, right if paddle is left of ball;
# no-op is preferred when aligned.
W3[0, 1] = 15.0
W3[0, 3] = -15.0
b3[0] = 5.0
b3[2] = -5.0
b3[4] = -5.0
b3[5] = -5.0
```