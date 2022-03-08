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

    # Compute intersection
    x_min = np.min([gt_box[0], prediction_box[0]])
    x_max = np.max([gt_box[2], prediction_box[2]])

    delta_x_gt = gt_box[2] - gt_box[0]
    delta_x_pred = prediction_box[2] - prediction_box[0]

    y_min = np.min([gt_box[1], prediction_box[1]])
    y_max = np.max([gt_box[3], prediction_box[3]])

    
    delta_y_gt = gt_box[3] - gt_box[1]
    delta_y_pred = prediction_box[3] - prediction_box[1]

    x_intersection = delta_x_gt +  delta_x_pred - (x_max - x_min)
    y_intersection = delta_y_gt +  delta_y_pred - (y_max - y_min)

    if x_intersection <=0 or y_intersection <=0:
        return 0

    intersection = x_intersection * y_intersection
    # Compute union
    union = delta_x_gt * delta_y_gt + delta_x_pred * delta_y_pred - intersection

    iou = intersection/union

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
    if num_tp + num_fp == 0:
        return 1

    return num_tp/(num_tp + num_fp)

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
    if num_tp + num_fn == 0:
        return 0

    return num_tp/(num_tp + num_fn)


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
    # Find all possible matches with a IoU >= iou threshold
    iou = np.array([[calculate_iou(pred, gt) for pred in prediction_boxes] for gt in gt_boxes])
    possible_matches = np.array(np.where(iou >= iou_threshold))
    iou_matches = np.array([[iou[mat[0],mat[1]], mat[0], mat[1]] for mat in possible_matches.T])
    sorted_iou = np.array(sorted(iou_matches, key=lambda x: -x[0]))

    # Find all matches with the highest IoU threshold
    predicted_indexes = []
    gt_indexes = []
    for [_, gt_index, pred_index] in sorted_iou:
        if pred_index in predicted_indexes or gt_index in gt_indexes:
            continue
        predicted_indexes.append(int(pred_index))
        gt_indexes.append(int(gt_index))
    
    # return np.array([]), np.array([])
    return np.array(prediction_boxes[predicted_indexes]), np.array(gt_boxes[gt_indexes])


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
    pred_matches, gt_matches = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)
    
    num_matches = pred_matches.shape[0]
    num_gt = gt_boxes.shape[0]
    num_pred = prediction_boxes.shape[0]

    return {"true_pos": num_matches, "false_pos": num_pred - num_matches, "false_neg": num_gt - num_matches}


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

    match_result = [calculate_individual_image_result(all_prediction_boxes[i], all_gt_boxes[i], iou_threshold) 
                    for i in range(len(all_prediction_boxes))]
    tp = 0
    fp = 0
    fn = 0
    for res in match_result:
        tp += res["true_pos"]
        fp += res["false_pos"]
        fn += res["false_neg"]
    
    precision = calculate_precision(tp, fp, fn)
    recall = calculate_recall(tp, fp, fn)
    return (precision, recall)


def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:list of np.array of floats): each element in the list
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
    num_images = len(all_prediction_boxes)
    precisions = []
    recalls = []
    for i in confidence_thresholds:

        pred_boxes = [all_prediction_boxes[image]
                    [np.array(np.where(confidence_scores[image] > i))[0],:] 
                    for image in range(num_images)]
        p, r = calculate_precision_recall_all_images(pred_boxes, all_gt_boxes, iou_threshold)
        precisions.append(p)
        recalls.append(r)
            
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
    num_recall_levels = 11
    recall_levels = np.linspace(0, 1.0, num_recall_levels)
    # YOUR CODE HERE

    # Sort precision and recall pairs according to accending recall value
    prec_rec_sorted = np.array(sorted([[precisions[i], recalls[i]] for i in range(len(recalls))], key=lambda x: x[1]))
    precisions_sorted = prec_rec_sorted[:,0]
    recalls_sorted = prec_rec_sorted[:,1]

    average_precision = 0

    recall_index = 0
    prec_max_right_index = np.argmax(precisions_sorted)
    prec_max_value = precisions_sorted[prec_max_right_index]
    precisions_linspace = [] 
    for i in recall_levels:
        # Finding the right recall value for the current interpolation point
        while recall_index < len(recalls_sorted) and recalls_sorted[recall_index] < i:
            recall_index += 1     

        # Finding the right precisiton value corresponding to the recall value
        if recall_index > prec_max_right_index: 
            sub_array = precisions_sorted[recall_index:]
            if len(sub_array) == 0:
                prec_max_right_index += 1
                prec_max_value = 0
            else:
                prec_max_right_index = np.argmax(np.array(sub_array)) + recall_index
                prec_max_value = precisions_sorted[prec_max_right_index]

        precisions_linspace.append(prec_max_value)

    average_precision = np.array(precisions_linspace).mean() 
    
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
