import math
import numpy

def sortPoints(points):
    points = points.tolist()
    # Start with one point. Doesn't matter which
    newPoints = [points.pop()]

    # Work through the list of points
    while points:
        # The next point in the polygon is the closest point to the last one
        lastPoint = newPoints[-1]

        closestPointIndex = None
        closestDistance = None
        for i, point in enumerate(points):
            # Pythagorean theorem
            distance = math.sqrt(abs(point[0] - lastPoint[0]) ** 2 + abs(point[1] - lastPoint[1]) ** 2)
            if closestDistance is None or distance < closestDistance:
                closestDistance = distance
                closestPointIndex = i
        # Now that we know the closest point, append to to the list
        newPoints.append(points.pop(closestPointIndex))
    newPoints.pop()
    newPoints = numpy.array(newPoints)
    return newPoints