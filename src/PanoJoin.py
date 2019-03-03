import argparse
import cv2
from points_of_intersection import common_points, _debug_poi
from stitch_image import stitch_images
import numpy as np

_SIZE = (1000, 1000)
_aspect_ratio = 0.125


def _resize_im(img, shape):
	identity_mat = np.float32([[1, 0, _SIZE[0]/4],[0, 1, _SIZE[1]/4]])
	return cv2.warpAffine(img, identity_mat, shape, flags=cv2.INTER_LINEAR)


def capture(vid_src):
	vid_cap = cv2.VideoCapture(vid_src)
	def nextFrame():
		ret, im = vid_cap.read()
		im = _resize_im(im, _SIZE)
		return ret, im
	return nextFrame

def src_seq(files):
	img_seq = [cv2.imread(file_name) for file_name in files]
	img_seq = [cv2.resize(img, (500, 250)) for img in img_seq]
	def nextFrame():
		if len(img_seq) == 0:
			return False, None
		im = img_seq.pop()
		im = _resize_im(im, _SIZE)
		return True, img_seq.pop()
	return nextFrame

def main(vid_src):
	if len(vid_src) == 1:
		cap = capture(vid_src[0])
		ret, img_frame = cap()
	else:
		cap = src_seq(vid_src)
	ret, prev_frame = cap()
	ret, new_frame = cap()
	result = prev_frame.copy()
	while ret:
		im_1 = result.copy()
		im_2 = new_frame.copy()
		pt_1, pt_2 = common_points(result, new_frame)
		_debug_poi(im_1, im_2)
		if len(pt_1) > 4:
			result = stitch_images(result, pt_1, new_frame, pt_2)
			cv2.imshow("Result", result)
			cv2.waitKey(33)
		prev_frame = new_frame
		ret, new_frame = cap()

if __name__ == '__main__':
	desc = "Panography is helpful to give us more information of the surroundings."
	parser = argparse.ArgumentParser(description = desc)
	parser.add_argument(
		'--file',
		action="store",
		dest="file_name",
		nargs='+',
		default= 0,
		help= "Use a video file in place of camera feed",
		required = True)
	main(parser.parse_args().file_name)
	cv2.destroyAllWindows()