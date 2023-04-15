import heapq

def heuristic(a, b):
    """
    Returns the Euclidean distance between points a and b.

    Args:
    a (tuple): A tuple representing the coordinates of the first point.
    b (tuple): A tuple representing the coordinates of the second point.

    Returns:
    The Euclidean distance between points a and b.
    """
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

def astar(array, start, goal):
    """
    Returns a list of tuples representing the path from the start point to the
    goal point in the given matrix, or None if no such path exists.

    Args:
    array (list of lists): The matrix to search in.
    start (tuple): A tuple representing the coordinates of the start point.
    goal (tuple): A tuple representing the coordinates of the goal point.

    Returns:
    A list of tuples representing the path from the start point to the goal point
    in the given matrix, or None if no such path exists.
    """

    neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]

    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []

    heapq.heappush(oheap, (fscore[start], start))

    while oheap:

        current = heapq.heappop(oheap)[1]

        if current == goal:
#            data = []
#            while current in came_from:
#                data.append(current)
#                current = came_from[current]
            return True #data[::-1]

        close_set.add(current)

        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)

            if 0 <= neighbor[0] < len(array):
                if 0 <= neighbor[1] < len(array[0]):
                    if array[neighbor[0]][neighbor[1]] == 0:
                        continue
                else:
                    # neighbor not in the horizontal bounds of the matrix
                    continue
            else:
                # neighbor not in the vertical bounds of the matrix
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))

    return None


#matrix = [
#    [1, 1, 1, 0],
#    [0, 0, 1, 0],
#    [1, 1, 1, 1],
#    [0, 0, 0, 1]
#]
#
#start = (0, 0)
#goal = (3, 3)
#
#path = astar(matrix, start, goal)
#print(path)