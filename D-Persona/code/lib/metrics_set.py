import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

def get_dice_threshold(output, mask, threshold):
    """
    :param output: output shape per image, float, (0,1)
    :param mask: mask shape per image, float, (0,1)
    :param threshold: the threshold to binarize output and feature (0,1)
    :return: dice of threshold t
    """
    smooth = 1e-6

    zero = torch.zeros_like(output)
    one = torch.ones_like(output)
    output = torch.where(output > threshold, one, zero)
    mask = torch.where(mask > threshold, one, zero)
    output = output.view(-1)
    mask = mask.view(-1)
    intersection = (output * mask).sum()
    dice = (2. * intersection + smooth) / (output.sum() + mask.sum() + smooth)

    return dice


def get_soft_dice(outputs, masks):
    """
    :param outputs: B * output shape per image
    :param masks: B * mask shape per image
    :return: average dice of B items
    """
    dice_list = []
    for this_item in range(outputs.size(0)):
        output = outputs[this_item]
        mask = masks[this_item]
        dice_item_thres_list = []
        for thres in [0.1, 0.3, 0.5, 0.7, 0.9]:
            dice_item_thres = get_dice_threshold(output, mask, thres)
            dice_item_thres_list.append(dice_item_thres.data)
        dice_item_thres_mean = np.mean(dice_item_thres_list)
        dice_list.append(dice_item_thres_mean)

    return np.mean(dice_list)


def get_iou_threshold(output, mask, threshold):
    """
    :param output: output shape per image, float, (0,1)
    :param mask: mask shape per image, float, (0,1)
    :param threshold: the threshold to binarize output and feature (0,1)
    :return: iou of threshold t
    """
    smooth = 1e-6

    zero = torch.zeros_like(output)
    one = torch.ones_like(output)
    output = torch.where(output > threshold, one, zero)
    mask = torch.where(mask > threshold, one, zero)

    intersection = (output * mask).sum()
    total = (output + mask).sum()
    union = total - intersection
    IoU = (intersection + smooth) / (union + smooth)

    return IoU


def get_soft_iou(outputs, masks):
    """
    :param outputs: B * output shape per image
    :param masks: B * mask shape per image
    :return: average iou of B items
    """
    iou_list = []
    for this_item in range(outputs.size(0)):
        output = outputs[this_item]
        mask = masks[this_item]
        iou_item_thres_list = []
        for thres in [0.1, 0.3, 0.5, 0.7, 0.9]:
            iou_item_thres = get_iou_threshold(output, mask, thres)
            iou_item_thres_list.append(iou_item_thres)
        iou_item_thres_mean = np.mean(iou_item_thres_list)
        iou_list.append(iou_item_thres_mean)

    return np.mean(iou_list)

# =========== GED ============= #


def segmentation_scores(mask1, mask2):
    IoU = get_iou_threshold(mask1, mask2, threshold=0.5)
    return 1.0 - IoU


def generalized_energy_distancex(label_list, pred_list):
    label_label_dist = [segmentation_scores(label_1, label_2) for i1, label_1 in enumerate(label_list)
                        for i2, label_2 in enumerate(label_list) if i1 != i2]
    pred_pred_dist = [segmentation_scores(pred_1, pred_2) for i1, pred_1 in enumerate(pred_list)
                      for i2, pred_2 in enumerate(pred_list) if i1 != i2]
    pred_label_list = [segmentation_scores(pred, label) for i, pred in enumerate(pred_list)
                       for j, label in enumerate(label_list)]
    GED = 2 * sum(pred_label_list) / len(pred_label_list) \
          - sum(label_label_dist) / len(label_label_dist) - sum(pred_pred_dist) / len(pred_pred_dist)
    return GED


def get_GED(batch_label_list, batch_pred_list):
    """
    :param batch_label_list: list_list
    :param batch_pred_list:
    :return:
    """
    batch_size = len(batch_pred_list)
    GED = 0.0
    for idx in range(batch_size):
        GED_temp = generalized_energy_distancex(label_list=batch_label_list[idx], pred_list=batch_pred_list[idx])
        GED = GED + GED_temp
    return GED / batch_size

def compute_dice_accuracy(label, mask):
    smooth = 1e-8
    batch = label.size(0)
    m1 = label.reshape(batch, -1).float()  # Flatten
    m2 = mask.reshape(batch, -1).float()  # Flatten
    intersection = (m1 * m2).sum().float()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def iou(x, y, axis=-1):
	smooth = 1e-8
	iou_ = ((x & y).sum(axis)) / ((x | y).sum(axis)+smooth)
	iou_[np.isnan(iou_)] = 1.
	return iou_

# exclude background
def distance(x, y):
	try:
		per_class_iou = iou(x[:, None], y[None, :], axis=-2)
	except MemoryError:
		per_class_iou = []
		for x_ in x:
			per_class_iou.append(iou(np.expand_dims(x_, axis=0), y[None, :], axis=-2))
		per_class_iou = np.concatenate(per_class_iou)
	return 1 - per_class_iou[..., 1:].mean(-1)


def calc_generalised_energy_distance(dist_0, dist_1, num_classes):
	dist_0 = dist_0.reshape((len(dist_0), -1))
	dist_1 = dist_1.reshape((len(dist_1), -1))
	dist_0 = dist_0.numpy().astype("int")
	dist_1 = dist_1.numpy().astype("int")

	eye = np.eye(num_classes)
	dist_0 = eye[dist_0].astype('bool')
	dist_1 = eye[dist_1].astype('bool')

	cross_distance = np.mean(distance(dist_0, dist_1))
	distance_0 = np.mean(distance(dist_0, dist_0))
	distance_1 = np.mean(distance(dist_1, dist_1))
	return cross_distance, distance_0, distance_1


# Metrics for Uncertainty
def generalized_energy_distance(labels, preds, thresh=0.5, num_classes=2):
	pred_masks = (preds > thresh).float()
	cross, d_0, d_1 = calc_generalised_energy_distance(labels[0], pred_masks[0], num_classes)
	GED = 2 * cross - d_0 - d_1

	return GED

def generalized_energy_distance_deep_seek(P, Q, distance_fn):
    """
    Compute the Generalized Energy Distance (GED) between two sets of samples.

    Args:
        P (torch.Tensor): Samples from distribution P, shape (n_samples_P, n_features).
        Q (torch.Tensor): Samples from distribution Q, shape (n_samples_Q, n_features).
        distance_fn (callable): A distance function (e.g., Euclidean distance).

    Returns:
        torch.Tensor: The GED value.
    """
    # Compute pairwise distances
    XX = -distance_fn(P, P)  # Distance between samples from P
    YY = -distance_fn(Q, Q)  # Distance between samples from Q
    XY = -distance_fn(P, Q)  # Distance between samples from P and Q

    # Compute the GED formula
    ged = 2 * torch.mean(XY) - torch.mean(XX) - torch.mean(YY)
    return ged

# Example distance function (Euclidean distance)
def euclidean_distance(X, Y):
    """
    Compute the pairwise Euclidean distance between two sets of samples.

    Args:
        X (torch.Tensor): Samples, shape (n_samples_X, n_features).
        Y (torch.Tensor): Samples, shape (n_samples_Y, n_features).

    Returns:
        torch.Tensor: Pairwise distances, shape (n_samples_X, n_samples_Y).
    """
    X_norm = (X**2).sum(dim=1, keepdim=True)  # ||X||^2
    Y_norm = (Y**2).sum(dim=1, keepdim=True)  # ||Y||^2
    distances = X_norm + Y_norm.T - 2 * X * Y
    distances = torch.sqrt(torch.clamp(distances, min=0))  # Ensure non-negative
    return distances

def iou_with_torch(x, y, axis=-1):
    smooth = 1e-8
    intersection = (x & y).sum(dim=axis)
    union = (x | y).sum(dim=axis) + smooth
    iou_ = intersection / union

    # Replace NaNs with 1
    iou_[torch.isnan(iou_)] = 1.0
    return iou_

# Exclude background
def distance_with_torch(x, y):
    try:
        per_class_iou = iou_with_torch(x[:, None], y[None, :], axis=-2)
    except RuntimeError:  # PyTorch raises RuntimeError instead of MemoryError
        per_class_iou = []
        for x_ in x:
            per_class_iou.append(iou_with_torch(x_.unsqueeze(0), y[None, :], axis=-2))
        per_class_iou = torch.cat(per_class_iou)

    return 1 - per_class_iou[..., 1:].mean(dim=-1)

# def calc_generalised_energy_distance_with_torsh(dist_0, dist_1, num_classes):
#     dist_0 = dist_0.reshape((len(dist_0), -1))
#     dist_1 = dist_1.reshape((len(dist_1), -1))

#     dist_0 = dist_0.to(torch.long)
#     dist_1 = dist_1.to(torch.long)

#     eye = torch.eye(num_classes, dtype=torch.bool, device=dist_0.device)
#     dist_0 = eye[dist_0]
#     dist_1 = eye[dist_1]

#     cross_distance = torch.mean(distance_with_torch(dist_0, dist_1))
#     distance_0 = torch.mean(distance_with_torch(dist_0, dist_0))
#     distance_1 = torch.mean(distance_with_torch(dist_1, dist_1))
#     return cross_distance, distance_0, distance_1

# # Metrics for Uncertainty
# def generalized_energy_distance_with_torsh(labels, preds, thresh=0.5, num_classes=2):
# 	pred_masks = torch.sigmoid(10 * (preds - thresh))
# 	cross, d_0, d_1 = calc_generalised_energy_distance_with_torsh(labels[0], pred_masks[0], num_classes)
# 	GED = 2 * cross - d_0 - d_1

# 	return GED

def dice_at_all(labels, preds, thresh=0.5, is_test=False):
    pred_masks = (preds > thresh).float()
    dice_each = []
    dice_max = []
    dice_max_reverse = []

    dice_matrix = np.zeros([labels.shape[1], pred_masks.shape[1]])
    for i in range(labels.shape[1]):
    	for j in range(pred_masks.shape[1]):
            dice_matrix[i,j] = compute_dice_accuracy(labels[:,i], pred_masks[:,j])

    dice_max = dice_matrix.max(0).mean() # many predictions
    dice_max_reverse = dice_matrix.max(1).mean() # many labels
	
    cost_matrix = 1 - dice_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    dice_match = []
    for i in range(labels.shape[1]):
        dice_match.append(1-cost_matrix[i, col_ind[i]])
        if is_test:
            dice_each.append(dice_matrix[i, i])
        else:
            dice_each.append(1-cost_matrix[i, col_ind[i]])
    dice_match = np.mean(dice_match)
    
    return dice_max, dice_max_reverse, dice_match, dice_each

def dice_at_thresh(labels, preds):
	thres_list = [0.1, 0.3, 0.5, 0.7, 0.9]

	pred_mean = preds.mean(1)
	label_mean = labels.mean(1).float()

	dice_scores = []
	for thresh in thres_list:
		pred_binary = (pred_mean > thresh).float()
		label_binary = (label_mean > thresh).float()
		dice_scores.append(compute_dice_accuracy(label_binary, pred_binary))
	dice_scores = np.mean(dice_scores)
	return dice_scores
