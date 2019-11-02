def IoU(bbox1, bbox2):
    """ Compute IoU of two bounding boxes

    Args:
        bbox1 - 4-tuple (x, y, w, h) where (x, y) is the top left corner of
            the bounding box, and (w, h) are width and height of the box.
        bbox2 - 4-tuple (x, y, w, h) where (x, y) is the top left corner of
            the bounding box, and (w, h) are width and height of the box.
    Returns:
        score - IoU score
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    score = 0
    ### YOUR CODE HERE

    yTop = max(y1, y2)
    yBottom = min(y1 + h1, y2 + h2)
    xLeft = max(x1, x2)
    xRight = min(x1 + w1, x2 + w2)
    if xRight < xLeft or yBottom < yTop:
        return 0.0
    intersection_area = (xRight - xLeft) * (yBottom - yTop)
    area1 = w1 * h1
    area2 = w2 * h2
    score = intersection_area / float(area1 + area2 - intersection_area)


    ### END YOUR CODE

    return score


