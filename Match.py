import cv2
import numpy as np

# 读取两张彩色图像
img1 = cv2.imread('5_real_image.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread('15_synthesized_image2.jpg', cv2.IMREAD_COLOR)
# 初始化SIFT检测器
# sift = cv2.SIFT_create()
sift = cv2.KAZE_create()

# 使用SIFT算子检测图片的关键点和描述符
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 使用FLANN匹配器进行特征点匹配
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# 选择好的匹配点
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# 获取匹配点的坐标
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 使用单应性矩阵计算变换矩阵
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# 绘制匹配点
for i, match in enumerate(good_matches):
    pt1 = tuple(map(int, kp1[match.queryIdx].pt))
    pt2 = tuple(map(int, kp2[match.trainIdx].pt))

    color = tuple(map(int, np.random.randint(0, 255, 3)))
    img1 = cv2.circle(img1, pt1, 5, color, -1)
    img2 = cv2.circle(img2, pt2, 5, color, -1)
    img = cv2.hconcat([img1, img2])
    cv2.line(img, pt1, (pt2[0] + 768, pt2[1]), color, 3)
    if i % 1 == 0:
        cv2.imshow("Matches", img)
        cv2.waitKey(0)

# 显示匹配结果
cv2.imshow("Matches", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
