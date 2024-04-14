import cv2
import numpy as np


def find_keypoints_and_descriptors(image):
    """
    Find keypoints and descriptors of the image using SIFT.
    """
    sift = cv2.SIFT_create()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_features(des1, des2):
    """
    Match features between two sets of descriptors using FLANN based matcher.
    """
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    return matches


def filter_matches(matches, ratio=0.75):
    """
    Apply Lowe's ratio test to filter out weak matches.
    """
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches


def find_homography(kp1, kp2, matches):
    """
    Compute homography matrix from matched features.
    """
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return M, mask


def stitch_images(img1, img2, M):
    """
    Stitch two images using the provided homography matrix.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners_img1_transformed = cv2.perspectiveTransform(corners_img1, M)
    corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    all_corners = np.concatenate((corners_img1_transformed, corners_img2), axis=0)
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    output_img = cv2.warpPerspective(img1, H_translation.dot(M), (x_max - x_min, y_max - y_min))
    output_img[translation_dist[1]:translation_dist[1] + h2, translation_dist[0]:translation_dist[0] + w2] = img2

    return output_img


# Main stitching logic
def create_panorama(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    # Read all frames from the video
    ret, frame = cap.read()
    while ret:
        frames.append(frame)
        ret, frame = cap.read()
    cap.release()

    panorama = frames[0]
    kp_last, des_last = find_keypoints_and_descriptors(frames[0])

    for i in range(1, len(frames)):

        if (i % 30 == 0):
            kp_current, des_current = find_keypoints_and_descriptors(frames[i])
            matches = match_features(des_last, des_current)
            good_matches = filter_matches(matches)

            if len(good_matches) > 10:
                M, _ = find_homography(kp_last, kp_current, good_matches)
                panorama = stitch_images(panorama, frames[i], M)
                # panorama1 = cv2.resize(panorama, (1280,860))
                #cv2.imshow('Panoramic View', panorama)

                # Update last_frame
                last_frame = panorama
                kp_last, des_last = find_keypoints_and_descriptors(last_frame)
            else:
                print("Not enough matches found - {}/{}".format(len(good_matches), 10))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    output_file_path = 'static/panoramas/panorama.jpg'
    cv2.imwrite(output_file_path, panorama)
    return "panorama.jpg"
