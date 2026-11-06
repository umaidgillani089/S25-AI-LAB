
import math

visited = 0

def alpha_beta(depth, node, is_max, scores, max_depth, alpha, beta):
    global visited

    if depth == max_depth:
        visited += 1
        return scores[node]

    if is_max:
        best = float('-inf')

        for i in range(2):
            result = alpha_beta(
                depth + 1, node * 2 + i, False,
                scores, max_depth, alpha, beta
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
                depth + 1, node * 2 + i, True,
                scores, max_depth, alpha, beta
            )

            best = min(best, result)
            beta = min(beta, best)

            if alpha >= beta:
                break

        return best


scores = [3, 5, 2, 9, 3, 5, 2, 9]

max_depth = int(math.log2(len(scores)))

answer = alpha_beta(
    0, 0, True, scores, max_depth,
    float('-inf'), float('inf')
)

print("Value:", answer)
print("Counts:", visited)
