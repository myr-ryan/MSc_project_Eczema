"""
This file is created by Zihao Wang, modified by Yingrui Ma
README

This file contains all necessary classes and functions for evaluating the segmentation network.
The generated (skin/AD) crops can then be used as the inputs for severity network.

The document of this project is available here:
    https://github.ic.ac.uk/tanaka-group/EczemaNet-DeepLearning-Segmentation/blob/master/README.md

"""

import matplotlib.path as pltPath
import numpy as np
import math
from matplotlib import pyplot as plt
import cv2


def draw_countours(img):
	"""
		This function is to draw and save contour images.

	"""
	boxes = []
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)[1]
	result = img.copy()
	contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = contours[0] if len(contours) == 2 else contours[1]
	for cntr in contours:
		rect = cv2.minAreaRect(cntr)
		# only returns rectangle!
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		area = cv2.contourArea(cntr)
		# Abandon boxes with too small area
		if(area > 10000.0):
			boxes.append(box)
			result = cv2.drawContours(result,[box],0,(0,0,255),5)
			# box_result = cv2.drawContours(box_result, cntr, 0, (255, 255, 255))
		
	plt.imshow(result)
	# plt.savefig('predicted.png', dpi=200)
	plt.show()

	return result, boxes



def compute_coverage_precision(true_mask, gt_boxes, pr_boxes):
	""" compute the precision, recall (coverage) and the F1-score for the given prediction.

	# Arguments
		true_mask: the ground truth mask for the given image
		gt_boxes: the labelled boxes of ground truth regions produced by border following
		pr_boxes: the predicted boxes of skin (or AD) regions

	# Returns
		return the metrics in the following order: recall, precision, f1, acc, dice_coef (which is the same as F1-score), [TP, TN, FP, FN]

	"""
	
	boxes_area = 0
	pixels_in_box_and_mask = 0
	pixels_in_mask = 0
	
	# coordinate transformation
	for box in gt_boxes:
		for point in box:
			point[1] = true_mask.shape[0] - point[1]

	for box in pr_boxes:
		for point in box:
			point[1] = true_mask.shape[0] - point[1]

	gt_area = 0
	pr_area = 0
	gt_neg = 0
	pr_neg = 0
	intersection_area = 0

	height = true_mask.shape[0]
	width = true_mask.shape[1]
	# print(true_mask.shape)
	# print(width)
	# print(height)


	gt_layer = np.zeros([height, width])
	pr_layer = np.zeros([height, width])
	gt_neg_layer = np.zeros([height, width])
	pr_neg_layer = np.zeros([height, width])
	# print(gt_layer.shape)

	# Loop through ground truth boxes
	for i in range(height):
		for j in range(width):
			for box in gt_boxes:
				path = pltPath.Path([box[0], box[1], box[2], box[3]])				
				if path.contains_points([[j,height - i]]):
					# print(j)
					if gt_layer[i, j] != 1.0:
						gt_layer[i, j] = 1.0
						gt_area += 1

	# Loop through predicted boxes
	for i in range(height):
		for j in range(width):
			for box in pr_boxes:
				path = pltPath.Path([box[0], box[1], box[2], box[3]])
				if path.contains_points([[j,height - i]]):
					if pr_layer[i, j] != 1.0:
						pr_layer[i, j] = 1.0
						pr_area += 1


	gt_neg_layer = np.ones([height, width]) - gt_layer
	pr_neg_layer = np.ones([height, width]) - pr_layer
	gt_neg = np.sum(gt_neg_layer)
	pr_neg = np.sum(pr_neg_layer)

	intersection_tp = np.logical_and(pr_layer, gt_layer)
	TP = np.sum(intersection_tp)

	intersection_tn = np.logical_and(pr_neg_layer, gt_neg_layer)
	TN = np.sum(intersection_tn)

	intersection_fp = np.logical_and(pr_layer, gt_neg_layer)
	FP = np.sum(intersection_fp)

	intersection_fn = np.logical_and(gt_layer, pr_neg_layer)
	FN = np.sum(intersection_fn)

	# Code for visualisation

	# # # print(gt_area)
	# # # print(pr_area)
	# gt_layer = gt_layer.astype(np.uint8) * 255
	# gt_layer = cv2.cvtColor(gt_layer, cv2.COLOR_BGR2RGB)
	# plt.imshow(gt_layer)
	# plt.show()
	# # plt.savefig('Gt_layer.png')
	# pr_layer = pr_layer.astype(np.uint8) * 255
	# pr_layer = cv2.cvtColor(pr_layer, cv2.COLOR_BGR2RGB)
	# plt.imshow(pr_layer)
	# plt.show()
	# # # # plt.savefig('Pr_layer.png')

	# pr_neg_layer = pr_neg_layer.astype(np.uint8) * 255
	# pr_neg_layer = cv2.cvtColor(pr_neg_layer, cv2.COLOR_BGR2RGB)
	# plt.imshow(pr_neg_layer)
	# plt.show()

	# gt_neg_layer = gt_neg_layer.astype(np.uint8) * 255
	# gt_neg_layer = cv2.cvtColor(gt_neg_layer, cv2.COLOR_BGR2RGB)
	# plt.imshow(gt_neg_layer)
	# plt.show()



	# intersection_tp = intersection_tp.astype(np.uint8) * 255
	# intersection_tp = cv2.cvtColor(intersection_tp, cv2.COLOR_BGR2RGB)
	# plt.imshow(intersection_tp)
	# plt.show()
	# # plt.savefig('inter_tp.png')

	# intersection_tn = intersection_tn.astype(np.uint8) * 255
	# intersection_tn = cv2.cvtColor(intersection_tn, cv2.COLOR_BGR2RGB)
	# plt.imshow(intersection_tn)
	# plt.show()
	# # plt.savefig('inter_tn.png', dpi=200)

	# intersection_fp = intersection_fp.astype(np.uint8) * 255
	# intersection_fp = cv2.cvtColor(intersection_fp, cv2.COLOR_BGR2RGB)
	# plt.imshow(intersection_fp)
	# plt.show()
	# # plt.savefig('inter_fp.png', dpi=200)

	# intersection_fn = intersection_fn.astype(np.uint8) * 255
	# intersection_fn = cv2.cvtColor(intersection_fn, cv2.COLOR_BGR2RGB)
	# plt.imshow(intersection_fn)
	# plt.show()
	# # plt.savefig('inter_fn.png', dpi=200)


	# print('TP is: ', TP)
	# print('TN is: ', TN)
	# print('FP is: ', FP)
	# print('FN is: ', FN)
	# print('total area: ', height*width)

	# define evaluation metrics
	coverage = 0 if gt_area==0 else TP / gt_area
	precision = 0 if pr_area==0 else TP / pr_area
	f1 = 0 if precision+coverage == 0 else 2 * precision * coverage / (precision + coverage)
	dice_coef = 0 if (TP+FP+TP+FN) == 0 else 2 * TP / (TP+FP+TP+FN)
	acc = 0 if (TP+TN+FP+FN) == 0 else (TP+TN)/(TP+TN+FP+FN)

	return coverage, precision, f1, acc, dice_coef, [TP, TN, FP, FN]


def compute_robustness_iou(true_mask, ref_boxes, perturbed_boxes):
	""" compute the IoU score for robustness analysis

	# Arguments
		true_mask: the ground truth mask for the given image
		ref_boxes: the labelled boxes for predictions on the unperturbed images
		perturbed_boxes: the labelled boxes for predictions on the perturbed images

	# Returns
		return the IoU score as a float number

		"""
	ref_region = np.zeros(shape=true_mask.shape[0:2], dtype=np.int8)
	perturbed_region = np.zeros(shape=true_mask.shape[0:2], dtype=np.int8)

	# coordinate transformation
	for box in ref_boxes:
		for point in box:
			point[1] = true_mask.shape[0] - point[1]
	for box in perturbed_boxes:
		for point in box:
			point[1] = true_mask.shape[0] - point[1]

	# iterate over all pixels to find which of them are contained in reference prediction
	# and which of them are contained in perturbed prediction
	for i in range(true_mask.shape[0]):
		for j in range(true_mask.shape[1]):
			# iterate through all the boxes to check if pixel is inside any of them
			for box in ref_boxes:
				path = pltPath.Path([box[0], box[1], box[2], box[3]])
				if path.contains_points([[j, true_mask.shape[0] - i]]):
					# count pixels in both ground truth mask and boxes (TP)
					ref_region[i, j] += 1
					break
			for box in perturbed_boxes:
				path = pltPath.Path([box[0], box[1], box[2], box[3]])
				if path.contains_points([[j, true_mask.shape[0] - i]]):
					# count pixels in both ground truth mask and boxes (TP)
					perturbed_region[i, j] += 1
					break

	intersection = np.multiply(ref_region, perturbed_region)
	union = np.add(ref_region, perturbed_region)
	# 	print("max: ", np.max(intersection))
	nb_intersection = np.sum([pixel > 0 for pixel in intersection])
	nb_union = np.sum([pixel > 0 for pixel in union])
	IoU = 0 if nb_union == 0 else nb_intersection / nb_union

	print("IoU: ", IoU)

	# Code for visualisation

	# 	plt.figure(dpi=200)
	# 	plt.tight_layout()

	# 	plt.subplot(141)
	# 	plt.box(False)
	# 	plt.axis('off')
	# 	plt.imshow(ref_region)
	# 	plt.title("ref pred")

	# 	plt.subplot(142)
	# 	plt.box(False)
	# 	plt.axis('off')
	# 	plt.imshow(perturbed_region)
	# 	plt.title("perturbed pred")

	# 	plt.subplot(143)
	# 	plt.box(False)
	# 	plt.axis('off')
	# 	plt.imshow(union)
	# 	plt.title("union")

	# 	plt.subplot(144)
	# 	plt.box(False)
	# 	plt.axis('off')
	# 	plt.imshow(intersection)
	# 	plt.title("intersection")

	# 	plt.savefig('test.eps', format='eps', bbox_inches='tight', pad_inches=0.2, dpi=200)
	# 	plt.show()

	return IoU

