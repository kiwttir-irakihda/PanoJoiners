import numpy as np
import cv2
from random import randint

feature_params = dict(maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7)
lk_params = dict(winSize  = (10, 10), 
	maxLevel = 5,
	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

_scalar = [(randint(0, 255),randint(0, 255),randint(0, 255)) for i in range(100)]

def _pt2Tup(pt):
	return pt[0][0],pt[0][1]

def _draw_circles(img, pts):
	for idx, pt_i in enumerate(pts):
		pt = _pt2Tup(pt_i)
		cv2.circle(img, pt, 10, _scalar[idx], -1)

def _filter_success(pts,n_pts, success, error):
	error = sorted(error, key= lambda x:x[0])
	n_err = len(error)
	th = error[n_err-1]
	pt_1 = [pts[idx] if (v[0] == 1 and error[idx] <= th) else None for idx, v in enumerate(success)]
	pt_2 = [n_pts[idx] if (v[0] == 1 and error[idx] <= th) else None for idx, v in enumerate(success)]
	pt_1 = list(filter(lambda x: x is not None, pt_1))
	pt_2 = list(filter(lambda x: x is not None, pt_2))
	return pt_1, pt_2

def common_points(img_1, img_2):
	g_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
	g_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
	pt_1 = cv2.goodFeaturesToTrack(g_1, mask = None, **feature_params)
	pt_2, std, err = cv2.calcOpticalFlowPyrLK(g_1, g_2, pt_1, None, **lk_params)
	pt_1, pt_2 = _filter_success(pt_1, pt_2, std, err)
	return pt_1, pt_2

def _debug_poi(img_1, img_2):
	pts, n_pts = common_points(img_1, img_2)
	_draw_circles(img_1, pts)
	_draw_circles(img_2, n_pts)
	cv2.imshow('img_1', img_1)
	cv2.imshow('img_2', img_2)
	return pts, n_pts

if __name__ == '__main__':
	img_1 = cv2.imread('../LunchRoom/LunchRoom/img01.jpg')
	img_2 = cv2.imread('../LunchRoom/LunchRoom/img02.jpg')
	_debug(img_1, img_2)
	cv2.destroyAllWindows()
