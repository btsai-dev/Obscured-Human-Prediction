def get_boxes_keypoint(keypoints):
    low_x = 999
    low_y = 999
    high_x = -999
    high_y = -999
    for coord in keypoints:
        if coord[0] > high_x and coord[2] > 0.05:
            high_x = coord[0]
        if coord[0] < low_x and coord[2] > 0.05:
            low_x = coord[0]
        if coord[1] > high_y and coord[2] > 0.05:
            high_y = coord[1]
        if coord[1] < low_y and coord[2] > 0.05:
            low_y = coord[1]

    return [low_x, low_y, high_x, high_y]