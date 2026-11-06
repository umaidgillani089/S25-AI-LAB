
import math

visited = 0

def minimax(depth, node, is_max, scores, target_depth):
    global visited

    if depth == target_depth:
        visited += 1
        return scores[node]

    left = minimax(
        depth + 1, node * 2, not is_max, scores, target_depth
    )

    right = minimax(
        depth + 1, node * 2 + 1, not is_max, scores, target_depth
    )

    if is_max:
        return max(left, right)
    else:
        return min(left, right)


scores = [3, 5, 2, 9, 0, 7]

tree_depth = math.ceil(math.log2(len(scores)))

scores.extend([0] * (2 ** tree_depth - len(scores)))

result = minimax(0, 0, True, scores, tree_depth)

print("Value:", result)
print("Count:", visited)
