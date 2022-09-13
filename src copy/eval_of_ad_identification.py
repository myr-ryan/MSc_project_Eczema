# This file is created by Zihao Wang, modified by Yingrui Ma.
import os
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models as sm
sm.set_framework('keras')
import ad_seg_utils as seg_utils
import bb_eval_utils as eval_utils
import csv
import math
import sys

def main():
	################################ Model loading (this part will be replaced with new data type soon) ################################
	SEGMENTATION = str(sys.argv[2])
	SUFFIX = str(sys.argv[4])
	MODEL_NAME = str(sys.argv[6])
	# Cropping quality of "AD" or "skin"
	EVAL_TYPE = str(sys.argv[8])

	print("Program initiating... type of segmentation: " + SEGMENTATION + ", type of dataset: " + SUFFIX)

	BIN_SEG = True
	CLASSES = ['background', 'skin']
	WEIGHTS = np.array([1, 1])
	target_idx = 1
	BACKBONE = 'efficientnetb3'
	preprocess_input = sm.get_preprocessing(BACKBONE)

	# set parameters based on the type of segmentation
	if SEGMENTATION == 'SKIN' or SEGMENTATION == 'skin':
		pass
	elif SEGMENTATION == 'AD' or SEGMENTATION == 'ad':
		BIN_SEG = False
		target_idx = 2
		CLASSES = ['background', 'skin', 'eczema']
		WEIGHTS = np.array([1, 1, 1])
	else:
		print('Unexpected type of segmentation, should be either skin or ad\n program terminated')
		return -1

	"""# Model Evaluation"""
	PROJ_DIR = "/rds/general/user/ym521/home/EczemaNet-DeepLearning-Segmentation-master"
	OUTPUT_DIR = os.path.join(PROJ_DIR, 'output_trained_on_S+T')
	PRED_DIR = os.path.join(OUTPUT_DIR, 'predictions/' + SEGMENTATION + '_' + SUFFIX)
	BB_DIR = os.path.join(OUTPUT_DIR, 'bounding_boxes')
	EVAL_DIR = os.path.join(OUTPUT_DIR, 'evaluations/' + SEGMENTATION + '_seg/cropping_quality/eval_' + EVAL_TYPE + '_region')
	MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
	DATA_DIR = os.path.join(PROJ_DIR, 'data')

	# new dataset
	x_test_dir = os.path.join(DATA_DIR, 'dataset_SWET/test_set_B_Kexin/reals')
	# x_test_dir = os.path.join(DATA_DIR, 'TLA4AE_origin')
	# y_test_dir = '/rds/general/user/ym521/home/Make_TLA_mask/output'
	y_test_dir = os.path.join(DATA_DIR, 'dataset_SWET/test_set_B_Kexin/labels')

	print('reading test images from: ' + str(x_test_dir))
	print('reading test masks from: ' + str(y_test_dir))

	test_dataset = seg_utils.Dataset(
		x_test_dir, 
		y_test_dir, 
		classes=CLASSES, 
		augmentation=None,
		preprocessing=seg_utils.get_preprocessing(preprocess_input),
		is_train=False,
		use_full_resolution=False,
		binary_seg=BIN_SEG,
	)
	eczema_dataset = seg_utils.Dataset(
		x_test_dir, 
		y_test_dir, 
		classes=['background', 'skin', 'eczema'],
		augmentation=None,
		preprocessing=seg_utils.get_preprocessing(preprocess_input),
		is_train=False,
		use_full_resolution=False,
		binary_seg=0,
	)

	model = seg_utils.load_model(dir=MODEL_DIR+MODEL_NAME, classes=CLASSES, weights=WEIGHTS)
	print('Trained model loaded!')

	################################ Mask prediction and evaluation ################################
	"""# Saving Masks Predictions"""
	# save all predictions 
	# clear previous predictions
	print('Creating directories and clearing previous masks...')
	os.system("mkdir -p " + PRED_DIR)
	os.system("rm " + PRED_DIR + "/*.jpg")
	os.system("rm " + PRED_DIR + "/*.JPG")
	os.system("rm " + BB_DIR + "/*.jpg")
	os.system("rm " + BB_DIR + "/*.JPG")
	os.system("rm " + EVAL_DIR + "/bb_evaluation_" + SEGMENTATION + "_" + SUFFIX + ".csv")
	print('Done! Now saving new prediction masks...')
	
	# Feb 8: export evaluation result as csv file 
	cov = []
	prec = []
	f1 = []
	acc = []
	dice = []

	with open(EVAL_DIR + '/bb_evaluation_' + SEGMENTATION + '_' + SUFFIX + '.csv', 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["file_name", "coverage", "precision", "f1_score", "accuracy", "dice_coef", "TP", "TN", "FP", "FN"])
		for i in range(len(test_dataset)):
			# save predicted masks
			image, _ = test_dataset[i]
			_, gt_mask = eczema_dataset[i]
			image = np.expand_dims(image, axis=0)
			pr_mask = model.predict(image)
			# change the last number to decide which mask to output. [0: background; 1: skin; 2: eczema]
			pr_img = pr_mask[0,:,:,target_idx]
			pr_img = (pr_img * 255).astype(np.uint8)
			cv2.imwrite(os.path.join(PRED_DIR, "pred_" + test_dataset.images_ids[i]), pr_img)
			# generate bounding boxes for each predicted mask
			# boxes = []
			pr_img = cv2.imread(os.path.join(PRED_DIR, "pred_" + test_dataset.images_ids[i]))
			pr_result, pr_boxes = eval_utils.draw_countours(pr_img)

			# For ground truth masks
			gt_mask = np.expand_dims(gt_mask, axis=0)
			if EVAL_TYPE == 'skin':
			# # For Cropping Quality of skin region
				gt_mask = gt_mask[0,:,:,2] + gt_mask[0,:,:,1]
			elif EVAL_TYPE == 'AD':
			# For Cropping Quality of AD region
				gt_mask = gt_mask[0,:,:,2]
			else:
				break
				print('Please check variable <EVAL_TYPE>')

			gt_mask = (gt_mask * 255).astype(np.uint8)
			cv2.imwrite(os.path.join(PRED_DIR, "mask_" + test_dataset.images_ids[i]), gt_mask)
			# generate bounding boxes for each ground truth mask
			gt_mask = cv2.imread(os.path.join(PRED_DIR, "mask_" + test_dataset.images_ids[i]))  
			gt_result, gt_boxes = eval_utils.draw_countours(gt_mask)

			# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			# thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)[1]
			# result = img.copy()
			# contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			# contours = contours[0] if len(contours) == 2 else contours[1]
			# for cntr in contours:
			# 	rect = cv2.minAreaRect(cntr)
			# 	box = cv2.boxPoints(rect)
			# 	box = np.int0(box)
			# 	area = cv2.contourArea(cntr)
			# 	# Abandon boxes with too small area
			# 	if(area > 10000.0):
			# 		boxes.append(box)
			# 		result = cv2.drawContours(result,[box],0,(0,0,255),2)

			# Feb 8: compute performance of bounding boxes
			coverage_per_image, precision_per_image, f1_per_image, acc_per_image, dice_per_image, [TP_per, TN_per, FP_per, FN_per] = eval_utils.compute_coverage_precision(gt_mask, gt_boxes, pr_boxes)

			cov.append(coverage_per_image)
			prec.append(precision_per_image)
			f1.append(f1_per_image)
			acc.append(acc_per_image)
			dice.append(dice_per_image)




			writer.writerow([test_dataset.images_ids[i], coverage_per_image, precision_per_image, f1_per_image, acc_per_image, dice_per_image, TP_per, TN_per, FP_per, FN_per])
			# save bounding boxes
			cv2.imwrite(os.path.join(BB_DIR, "bb_pred_" + test_dataset.images_ids[i]), pr_result)
			cv2.imwrite(os.path.join(BB_DIR, "bb_mask_" + test_dataset.images_ids[i]), gt_result)

		# Append the mean performance to the end of csv
		writer.writerow(['mean', np.mean(cov), np.mean(prec), np.mean(f1), np.mean(acc), np.mean(dice)])
		writer.writerow(['se', np.std(cov) / np.sqrt(len(cov)), np.std(prec) / np.sqrt(len(prec)), np.std(f1) / np.sqrt(len(f1)), np.std(acc) / np.sqrt(len(acc)), np.std(dice) / np.sqrt(len(dice))])

		print('Done!')

if __name__ == "__main__":
	main()