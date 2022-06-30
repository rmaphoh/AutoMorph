"""
===========================
@Author  : Linbo<linbo.me>
@Version: 1.0    25/10/2014
This is the implementation of the 
Zhang-Suen Thinning Algorithm for skeletonization.

Code taken from:
 https://github.com/linbojin/Skeletonization-by-Zhang-Suen-Thinning-Algorithm/
===========================
"""


def neighbours(x, y, image):
    "Return 8-neighbours of image point p1(x,y), in a clockwise order"
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    return [img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],     # p2,p3,p4,p5
            img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]    # p6,p7,p8,p9


def transitions(neighbours):
    "No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence"
    n = neighbours + neighbours[0:1]      # p2, p3, ... , p8, p9, p2
    return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )  # (p2,p3), (p3,p4), ... , (p8,p9), (p9,p2)


def thinning_zhang_suen(image):
    "the Zhang-Suen Thinning Algorithm"
    image_thinned = image.copy()  # deepcopy to protect the original image
    changing1 = changing2 = 1        # the points to be removed (set as 0)
    while changing1 or changing2:   # iterates until no further changes occur in the image
        # Step 1
        changing1 = []
        rows, columns = image_thinned.shape               # x for rows, y for columns
        for x in range(1, rows - 1):                     # No. of  rows
            for y in range(1, columns - 1):            # No. of columns
                p2, p3, p4, p5, p6, p7, p8, p9 = n = neighbours(x, y, image_thinned)
                if (image_thinned[x][y] == 1 and    # Condition 0: Point p1 in the object regions
                    2 <= sum(n) <= 6 and    # Condition 1: 2<= N(p1) <= 6
                    transitions(n) == 1 and    # Condition 2: S(p1)=1
                    p2 * p4 * p6 == 0 and    # Condition 3
                    p4 * p6 * p8 == 0):         # Condition 4
                        changing1.append((x, y))
        for x, y in changing1:
            image_thinned[x][y] = 0
        # Step 2
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                p2, p3, p4, p5, p6, p7, p8, p9 = n = neighbours(x, y, image_thinned)
                if (image_thinned[x][y] == 1 and        # Condition 0
                    2 <= sum(n) <= 6 and       # Condition 1
                    transitions(n) == 1 and      # Condition 2
                    p2 * p4 * p8 == 0 and       # Condition 3
                    p2 * p6 * p8 == 0):            # Condition 4
                        changing2.append((x, y))
        for x, y in changing2:
            image_thinned[x][y] = 0
    return image_thinned
