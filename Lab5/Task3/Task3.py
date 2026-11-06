
import math

count = 0

def minimax(depth, index, is_max, scores, target):
    global count

    if depth == target:
        count += 1
        return scores[index]

    if is_max:
        best = float('-inf')

        for i in range(2):
            result = minimax(
                depth + 1, index * 2 + i,
                False, scores, target
            )
            best = max(best, result)

        return best

    else:
        best = float('inf')

        for i in range(2):
            result = minimax(
                depth + 1, index * 2 + i,
                True, scores, target
            )
            best = min(best, result)

        return best


def alpha_beta(depth, index, is_max, scores, target, alpha, beta):
    global count

    if depth == target:
        count += 1
        return scores[index]

    if is_max:
        best = float('-inf')

        for i in range(2):
            result = alpha_beta(
                depth + 1, index * 2 + i, False,
                scores, target, alpha, beta
            )

            best = max(best, result)
            alpha = max(alpha, best)

            if alpha >= beta:
                break

        return best

    else:
        best = float('inf')

        for i in range(2):
            result = alpha_beta(
                depth + 1, index * 2 + i, True,
                scores, target, alpha, beta
            )

            best = min(best, result)
            beta = min(beta, best)

            if alpha >= beta:
                break

        return best


scores = [3, 5, 2, 9, 3, 5, 2, 9]
depth = int(math.log2(len(scores)))

count = 0
result = minimax(0, 0, True, scores, depth)
print("Minimax Value:", result)
print("Minimax Counts:", count)

count = 0
result = alpha_beta(
    0, 0, True, scores, depth,
    float('-inf'), float('inf')
)
print("AlphaBeta Value:", result)
print("AlphaBeta Counts:", count)


scores = [5, 3, 9, 2, 5, 3, 9, 2]

count = 0
result = alpha_beta(
    0, 0, True, scores, depth,
    float('-inf'), float('inf')
)

print("Reordered AlphaBeta Value:", result)
print("Reordered AlphaBeta Counts:", count)