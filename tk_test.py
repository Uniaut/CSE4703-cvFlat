import cv2
import numpy as np


def line_intersection(line1, line2, polar=False):
    if not polar:
        line1 = line1.reshape(2, 2)
        line2 = line2.reshape(2, 2)
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            return -1, -1
        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y
    else:
        rho1, theta1 = line1[0]
        rho2, theta2 = line2[0]
        A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
        b = np.array([[rho1], [rho2]])
        try:
            x0, y0 = np.linalg.solve(A, b)
        except:
            return [-1, -1]
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return [x0, y0]


def process(image: cv2.Mat):
    lsd = cv2.createLineSegmentDetector(0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lines = lsd.detect(image)[0]
    drawn_image = lsd.drawSegments(image, lines)
    cv2.imshow('draw', drawn_image)

    edges = cv2.Canny(image, 150, 200)
    cv2.imshow('edges', edges)

    mask = np.zeros((image.shape[0] + 2, image.shape[1] + 2), np.uint8)
    arbitrary = 150  # any value except 255 because edges are 255
    flooded = cv2.floodFill(edges, mask, (0, 0), arbitrary)[1]
    cv2.imshow('flooded', flooded)

    flooded_mask = np.zeros((flooded.shape))
    flooded_mask[edges == arbitrary] = 255
    cv2.imshow('flooded_mask', flooded_mask)

    segmented_edges = cv2.Canny(np.uint8(flooded_mask), 100, 200, L2gradient=True)
    cv2.imshow('segmented_edges', segmented_edges)

    lsd = cv2.createLineSegmentDetector(0)
    plane_ = np.zeros((segmented_edges.shape))
    lines = lsd.detect(segmented_edges.astype('uint8'))[0]
    drawn_image = lsd.drawSegments(segmented_edges, lines)
    cv2.imshow('drawn_image', drawn_image)

    # 밑에 코드에서 오류 발생해서 주석처리
    """
    for l in range(len(lines)):
        xy = line_intersection(lines[l], lines[l-1], False)
        if xy[0]<0 or xy[1]<0 or xy[0]>plane_.shape[0] or xy[1]>plane_.shape[1]:
            continue
    #     print(xy)
        plane_ = cv2.circle(drawn_image, xy, 5, (0, 0, 255), 2)
    cv2.imshow("plane_", plane_)
    cv2.waitKey(0)
    """

    plane = np.zeros(image.shape)
    lines = cv2.HoughLines(
        segmented_edges, 0.77, np.pi / 183, 100, None, 0, 0
    )  # hyper-parameter tuned using grid search then manually!
    # Draw the lines
    if lines is not None:
        for i in range(len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(plane, pt1, pt2, (arbitrary, 234, arbitrary), 2, cv2.LINE_AA)
    cv2.imshow('plane', plane)

    for l in range(len(lines)):
        xy = tuple(line_intersection(lines[l], lines[l - 1], polar=True))
        if xy[0] < 0 or xy[1] < 0 or xy[0] > plane.shape[0] or xy[1] > plane.shape[1]:
            continue
        plane = cv2.circle(plane, xy, 10, 255, 2)
    cv2.imshow('plane2', plane)

    points_ = []
    for l in range(len(lines)):
        xy = tuple(line_intersection(lines[l], lines[l - 1], polar=True))
        if xy[0] < 0 or xy[1] < 0 or xy[0] > plane.shape[0] or xy[1] > plane.shape[1]:
            continue
        points_.append(xy)

    src_points = np.zeros((4, 2))
    j = 0
    for idx, p in enumerate(points_):
        if not (idx == 3 or idx == 4):
            src_points[j, 0] = p[0]
            src_points[j, 1] = p[1]
            j += 1
    print(src_points)
    dst_points = np.zeros((4, 2))
    dst_points[0, 0] = image.shape[0]
    dst_points[0, 1] = 0
    dst_points[1, 0] = 0
    dst_points[1, 1] = 0
    dst_points[2, 0] = image.shape[0]
    dst_points[2, 1] = image.shape[1]
    dst_points[3, 0] = 0
    dst_points[3, 1] = image.shape[1]
    print(dst_points)

    h, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC)
    im1reg = cv2.warpPerspective(image, h, image.shape)

    cv2.imshow('im1reg', im1reg)
