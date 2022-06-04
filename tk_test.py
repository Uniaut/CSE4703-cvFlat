import itertools
import cv2
import numpy as np
import math


def _line_intersection(line1, line2, polar=False):
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


def get_ui_image(image: cv2.Mat):
    ui_image = image.copy()
    ui_image = cv2.circle(ui_image, ((image.shape[1] // 2), (image.shape[0] // 2)), 15, (255, 0, 255))
    return ui_image


def get_edge_image(image: cv2.Mat):
    image = image.copy()
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    blurred_split = cv2.split(blurred)
    edges_split = tuple(map(lambda img: cv2.Canny(img, 50, 200), blurred_split))
    cv2.imshow('Merged RGB', cv2.merge(edges_split))
    edges = sum(edges_split)
    return edges


def get_flood_edges(image: cv2.Mat):
    edges = image.copy()
    mask = np.zeros((image.shape[0] + 2, image.shape[1] + 2), np.uint8)
    # edges = cv2.dilate(image, np.ones((3, 3), np.uint8), iterations=2)
    arbitrary = 150  # any value except 255 because edges are 255

    for dx, dy in itertools.product([-10, 0, 10], [-10, 0, 10]):
        starting_point = ((edges.shape[1] // 2) + dx, (edges.shape[0] // 2) + dy)
        flooded = cv2.floodFill(edges, mask, starting_point, arbitrary)[1]
    
    flooded_ = flooded.copy()
    for dx, dy in itertools.product([-10, 0, 10], [-10, 0, 10]):
        starting_point = ((edges.shape[1] // 2) + dx, (edges.shape[0] // 2) + dy)
        flooded_ = cv2.circle(flooded_, starting_point, 3, (0, 0, 255))
    cv2.imshow('flooded', flooded_)

    flooded_mask = np.zeros((flooded.shape))
    flooded_mask[edges == arbitrary] = 255

    segmented_edges = cv2.Canny(np.uint8(flooded_mask), 100, 200, L2gradient=True)
    return segmented_edges


def show_hough_lines(image, lines, name):
    plane = np.zeros(image.shape)
    for i in range(len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv2.line(plane, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow(name, plane)


def process(image: cv2.Mat):
    # UI image
    ui_image = get_ui_image(image)
    cv2.imshow('1. original', ui_image)

    # Canny by rgb
    edges = get_edge_image(image)
    cv2.imshow('2. edges', edges)

    # segment center objects by flooding
    segmented_edges = get_flood_edges(edges)
    cv2.imshow('3. flood center & segment', segmented_edges)

    plane = np.zeros(image.shape)
    lines = cv2.HoughLines(
        segmented_edges, 1, np.pi / 180, 100, None, 0, 0
    )  # hyper-parameter tuned using grid search then manually!
    show_hough_lines(image, lines, 'original')

    '''
    for all lines sorted by -rho
    '''
    lines_sorted = sorted(list(lines), key=lambda l: l[0][0])
    # print(lines_sorted)

    lines_dict = dict()
    for i in range(len(lines_sorted)):
        rho = lines_sorted[i][0][0]
        rho = round(rho * 1 / (image.shape[0]), 0)
        theta = lines_sorted[i][0][1]
        theta = round(theta * 4 / (2 * np.pi), 0)
        # print('Value: ', rho, theta)
        if lines_dict.get((rho, theta)) is None:
            lines_dict[rho, theta] = lines_sorted[i]
    else:
        lines = list(lines_dict.values())

    show_hough_lines(image, lines, 'compressed')

    if len(lines) != 4:
        print('number of lines', len(lines))
        for a, b in lines_dict.items():
            print(a)
        return
    

    # line_intersection
    points_ = []
    for l1, l2 in itertools.combinations(lines, r=2):
        xy = tuple(_line_intersection(l1, l2, polar=True))
        if xy[0] < 0 or xy[1] < 0 or xy[0] > plane.shape[1] or xy[1] > plane.shape[0]:
            continue
        points_.append(xy)
        plane = cv2.circle(plane, xy, 10, 255, 2)
    cv2.imshow('plane2', plane)

    '''
    top-left, top-right, bottom-left, botton-right 정하기
    '''
    src_points = np.float32([[0, 0], [0, 0], [0, 0], [0, 0]])
    xsort = sorted(points_, key=lambda p:p[0])
    ysort = sorted(points_, key=lambda p:p[1])
    point_align = {
        (True, True): 0,
        (False, True): 1,
        (True, False): 2,
        (False, False): 3,
    }
    for p in points_:
        flag_x, flag_y = p in xsort[:2], p in ysort[:2]
        src_points[point_align[flag_x, flag_y]] = p

    dst_size = 500
    dst_points = np.float32([[0, 0], [dst_size, 0], [0, 1.414 * dst_size], [dst_size, 1.414 * dst_size]])

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    result = cv2.warpPerspective(image, matrix, (dst_size, int(1.414 * dst_size)))

    cv2.imshow('im1reg', result)
