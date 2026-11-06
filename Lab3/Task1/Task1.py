
from collections import deque

graph = {
    "A": ["B", "C"],
    "B": ["D", "E"],
    "C": ["F"],
    "D": [],
    "E": ["G"],
    "F": [],
    "G": []
}


def breadth_first_search(start, goal):
    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        current, path = queue.popleft()

        if current == goal:
            return path

        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None


def depth_first_search(start, goal):
    stack = [(start, [start])]
    visited = set()

    while stack:
        current, path = stack.pop()

        if current == goal:
            return path

        if current in visited:
            continue

        visited.add(current)

        for neighbor in reversed(graph[current]):
            stack.append((neighbor, path + [neighbor]))

    return None


print("BFS Path:", breadth_first_search("A", "G"))
print("DFS Path:", depth_first_search("A", "G"))
