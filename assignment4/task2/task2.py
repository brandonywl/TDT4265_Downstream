import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE

    # Define a function to get area to better explain the code
    # Gets an area by coordinates
    def get_area_coord(x, y):
        return x * y

    # Gets the area of a box
    def get_area(box):
        x_len = box[2] - box[0]
        y_len = box[3] - box[1]
        return get_area_coord(x_len, y_len)
    
    """
    With the defition of prediction_box and gt_box in args, we see the following: 
    
    [0] = x coordinate of the left most of box
    [1] = y coordinate of the bottom most of box
    [2] = x coordinate of the right most of box
    [3] = y coordinate of the top most of box

    Hence with the definition of x_offset and y_offset laid in my report, we can get the length of the overlap of x and y between the two boxes.
    """
    def get_intersection_area(prediction_box, gt_box):
        x_offset = max(0, min(prediction_box[2], gt_box[2]) - max(prediction_box[0], gt_box[0]))
        y_offset = max(0, min(prediction_box[3], gt_box[3]) - max(prediction_box[1], gt_box[1]))

        return get_area_coord(x_offset, y_offset)

    def get_union_area(prediction_box, gt_box, intersection_area):
        box1_area = get_area(prediction_box)
        box2_area = get_area(gt_box)
        return box1_area + box2_area - intersection_area




    # Compute intersection

    intersection = get_intersection_area(prediction_box, gt_box)
    union = get_union_area(prediction_box, gt_box, intersection)

    iou = intersection / union
    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    
    # Precision = TP / (TP + FP)
    base = num_tp + num_fp
    # Prevent edge case of DivideByZeroError
    if base == 0:
        return 1

    return num_tp / base
    



def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    
    # Recall = TP / (TP + FN)
    base = num_tp + num_fn
    # Prevent edge case of DivideByZeroError
    if base == 0:
        return 0

    return num_tp / base


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    # Get all IOUs and matches with IOU >= iou treshold
    matches = []
    for idx_i, gt_box in enumerate(gt_boxes):
        for idx_j, prediction_box in enumerate(prediction_boxes):
            iou = calculate_iou(prediction_box, gt_box)
            
            if iou >= iou_threshold:
                matches.append((iou, (idx_i, idx_j), prediction_box, gt_box))

    # Sort all matches on IoU in descending order
    matches.sort(reverse=True)

    # Find all matches with the highest IoU threshold

    def non_maximum_supression(matches):
        gt_indexes = []
        prediction_indexes = []
        prediction_box_matches = []
        gt_box_matches = []

        for match in matches:
            gt_idx, prediction_idx = match[1]
            prediction_box = match[2]
            gt_box = match[3]

            # As we are working in sorted order that is filtered, we just need to take the first prediction
            if gt_idx not in gt_indexes and prediction_idx not in prediction_indexes:
                gt_indexes.append(gt_idx)
                prediction_indexes.append(prediction_idx)
                prediction_box_matches.append(prediction_box)
                gt_box_matches.append(gt_box)
                
        return prediction_box_matches, gt_box_matches

    prediction_matches, gt_matches = non_maximum_supression(matches)

    return np.array(prediction_matches), np.array(gt_matches)


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """

    # Prediction boxes is all the predicted bboxes from the model less those that dont make the cut for the model already.
    matched_pred_boxes, _ = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)
    true_pos = len(matched_pred_boxes) # One entry == a match >= IoU treshold to any GT box, hence a true positive
    false_pos = len(prediction_boxes) - true_pos # False pos = Detected Pos - true pos
    false_neg = len(gt_boxes) - true_pos # All ground truths not detected = gt - true_pos

    return {
        "true_pos": true_pos,
        "false_pos": false_pos,
        "false_neg": false_neg
    }


def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    
    image_set = zip(all_prediction_boxes, all_gt_boxes)

    tp = fp = fn = 0

    for image in image_set:
        result = calculate_individual_image_result(image[0], image[1], iou_threshold)
        tp += result["true_pos"]
        fp += result["false_pos"]
        fn += result["false_neg"]

    precision = calculate_precision(tp, fp, fn)
    recall = calculate_recall(tp, fp, fn)

    return precision, recall


def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)
    # YOUR CODE HERE

    precisions = [] 
    recalls = []

    for confidence_threshold in confidence_thresholds:

        image_set = zip(all_prediction_boxes, confidence_scores)
        filtered_images = []

        for prediction_boxes, scores in image_set:
            box_set = zip(prediction_boxes, scores)
            filtered_boxes = []

            for box in box_set:
                if box[1] >= confidence_threshold:
                    filtered_boxes.append(box[0])
            
            filtered_images.append(filtered_boxes)

        precision, recall = calculate_precision_recall_all_images(filtered_images, all_gt_boxes, iou_threshold)
        precisions.append(precision)
        recalls.append(recall)

    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)
    # YOUR CODE HERE
    
    # First interpolate the curve
    # Following slide 49 of the lecture notes, for the point r, we take the maximum precision for R > r + 1
    # From the slides 28 and 29, the intuition is that a lower confidence interval -> higher recall, lower precision. Higher CI -> lower recall, higher precision
    # As the recall curve is computed in increasing order of confidence interval, we should expect an increase in recall and a fall in confidence in the order.
    # Hence we can work backwards by reversing precision and recall lists
    interpolations = []

    rev_precisions = precisions[::-1]
    rev_recalls = recalls[::-1]

    for idx, recall in enumerate(rev_recalls):
        max_precision = np.max(rev_precisions[idx:])
        if idx != 0 and recall == rev_recalls[idx - 1] and max_precision < interpolations[idx - 1]:
            interpolations.append(interpolations[idx - 1])
        else:
            interpolations.append(max_precision)


    # Next we compute the AUC of the curve at different recall levels
    precision_cumsum = 0
    for recall_level in recall_levels:
        precision = 0
        for idx, recall in enumerate(rev_recalls):
            if recall_level <= recall:
                precision = rev_precisions[idx]
                break
        precision_cumsum += precision


    average_precision = precision_cumsum / len(recall_levels)
    return average_precision


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
