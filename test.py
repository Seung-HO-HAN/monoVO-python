import numpy as np 
import cv2

from visual_odometry import PinholeCamera, VisualOdometry

num_img = 250

cam = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)
vo = VisualOdometry(cam, './dataset/poses/00.txt')

traj = np.zeros((400,400,3), dtype=np.uint8)

for img_id in xrange(num_img):
	img = cv2.imread('./dataset/sequences/00/image_0/'+str(img_id).zfill(6)+'.png', 0)

	vo.update(img, img_id)

	cur_t = vo.cur_t
	if(img_id > 2):
		x, y, z = cur_t[0], cur_t[1], cur_t[2]
	else:
		x, y, z = 0., 0., 0.
	draw_x, draw_y = int(x)+200, int(z)+60
	true_x, true_y = int(vo.trueX)+400, int(vo.trueZ)+60

	cv2.circle(traj, (draw_x,draw_y), 1, (img_id*255/num_img,255-img_id*255/num_img,0), 1)
	cv2.circle(traj, (true_x,true_y), 1, (0,0,255), 2)
	cv2.rectangle(traj, (10, 20), (400, 60), (0,0,0), -1)
	text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x,y,z)
	cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)

	cv2.imshow('Road facing camera', img)
	cv2.imshow('Trajectory', traj)
	cv2.waitKey(1)

cv2.imwrite('map.png', traj)
