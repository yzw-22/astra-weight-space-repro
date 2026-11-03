The network uses a single signed comparison: first-layer unit 0 computes `ball_column - paddle_column` from the one-hot paddle and ball channels. The second layer amplifies that signal, and the output layer maps positive values to a strong `right` logit, negative values to a strong `left` logit, and near-zero values to `no-op`. Unused actions are left at zero logits. This makes the paddle track the ball horizontally and stay still once aligned.

```edits
# Unit 0 of h1: ball column - paddle column.
# Channel 3 is the ball, channel 0 is the paddle.
W1[9:600:60, 0] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
W1[15:600:60, 0] = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
W1[21:600:60, 0] = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
W1[27:600:60, 0] = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
W1[33:600:60, 0] = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
W1[39:600:60, 0] = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
W1[45:600:60, 0] = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
W1[51:600:60, 0] = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
W1[57:600:60, 0] = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9]

W1[6:600:60, 0] = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
W1[12:600:60, 0] = [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2]
W1[18:600:60, 0] = [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3]
W1[24:600:60, 0] = [-4, -4, -4, -4, -4, -4, -4, -4, -4, -4]
W1[30:600:60, 0] = [-5, -5, -5, -5, -5, -5, -5, -5, -5, -5]
W1[36:600:60, 0] = [-6, -6, -6, -6, -6, -6, -6, -6, -6, -6]
W1[42:600:60, 0] = [-7, -7, -7, -7, -7, -7, -7, -7, -7, -7]
W1[48:600:60, 0] = [-8, -8, -8, -8, -8, -8, -8, -8, -8, -8]
W1[54:600:60, 0] = [-9, -9, -9, -9, -9, -9, -9, -9, -9, -9]

# Amplify the signed difference into h2.
W2[0, 0] = 2

# h2[0] > 0 => ball is right of paddle => right action.
# h2[0] < 0 => ball is left of paddle => left action.
# h2[0] = 0 => aligned => no-op wins.
W3[0, 3] = 10
W3[0, 1] = -10
b3[0] = 5
```