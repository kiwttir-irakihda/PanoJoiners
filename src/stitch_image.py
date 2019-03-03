import math
import numpy as np
import cv2
from points_of_intersection import common_points, _debug_poi

def _pt2Tup(pt):
	"""
	return: tuple y, x

	Converts numpy points to coordinates
	"""

	return pt[0][0],pt[0][1]

def _perspective_transform(row, col, h):
	x_1  = h[0][0]*col + h[0][1]*row + h[0][2]
	y_1  = h[1][0]*col + h[1][1]*row + h[1][2]
	norm = h[2][0]*col + h[2][1]*row + h[2][2]
	return int(y_1/norm) , int(x_1/norm)

def _stitch_affine(img_1, pt_1, img_2, pt_2):
	per_map, mask = cv2.findHomography(np.array(pt_2), np.array(pt_1), cv2.LMEDS,5.0)
	rows, cols, ch = img_2.shape
	result = cv2.warpPerspective(img_2, per_map, img_2.shape[1::-1])
	result = cv2.addWeighted(img_1,0.75,result,0.25,0)

	return result

def _slope(pt_1, pt_2):
	"""
	return int: slope between points

	Given two points compute slope
	"""

	r_1, c_1 = _pt2Tup(pt_1)
	r_2, c_2 = _pt2Tup(pt_2)
	return ((r_1 - r_2 + 0.0000001)/(c_1 - c_2 + 0.0000001))

def _get_rotate_deg(pt_1, pt_2):
	"""
	return img: warp map

	Tells by how much we must rotate img_2 to paste on img_1
	"""

	n_pt = len(pt_1)
	m_1 = _slope(pt_1[0], pt_1[1])
	m_2 = _slope(pt_2[0], pt_2[1])
	return math.degrees((math.atan(m_2) - math.atan(m_1)))

def _get_translate(pt_1, pt_2):
	"""
	return img: warp map

	Tells by how much we must move img_2 to paste on img_1
	"""

	r_1, c_1 = _pt2Tup(pt_1[0])
	r_2, c_2 = _pt2Tup(pt_2[0])
	d_r = r_1 - r_2
	d_c = c_1 - c_2
	return d_r, d_c


def _stitch_tr(img_1, pt_1, img_2, pt_2):
	"""
	return img: warp map

	Stitches the images by translating and rotating
	"""
	
	angle = _get_rotate_deg(pt_1, pt_2)
	d_r, d_c = _get_translate(pt_1, pt_2)
	center = tuple(np.array(img_1.shape[1::-1]) / 2)
	rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
	tra_mat = np.float32([[1, 0, d_c],[0, 1, d_r]])
	result = cv2.warpAffine(img_2, rot_mat, img_2.shape[1::-1], flags=cv2.INTER_LINEAR)
	result = cv2.warpAffine(result, tra_mat, result.shape[1::-1], flags=cv2.INTER_LINEAR)
	result = cv2.addWeighted(img_1,0.7,result,0.3,0)
	return result

def stitch_images(img_1, pt_1, img_2, pt_2):
	return _stitch_affine(img_1, pt_1, img_2, pt_2)

def _debug(img_1, pt_1, img_2, pt_2):
	res = stitch_images(img_1, pt_1, img_2, pt_2)
	cv2.imshow('res', res)
	cv2.waitKey(0)
	return res

if __name__ == '__main__':
	img_1 = cv2.imread('../LunchRoom/LunchRoom/img01.jpg')
	img_2 = cv2.imread('../LunchRoom/LunchRoom/img02.jpg')
	pt_1, pt_2 = _debug_poi(img_1, img_2)
	res = _debug(img_1, pt_1, img_2, pt_2)
	cv2.destroyAllWindows()
