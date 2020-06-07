# References
# https://web.archive.org/web/20140115053733/http://cs.bath.ac.uk:80/brown/papers/ijcv2007.pdf
# https://github.com/linrl3/Image-Stitching-OpenCV/blob/master/Image_Stitching.py
# https://living-sun.com/pt/python/708914-how-do-you-compute-a-homography-given-two-poses-python-opencv-homography.html
# https://towardsdatascience.com/image-panorama-stitching-with-opencv-2402bde6b46c
# https://towardsdatascience.com/image-stitching-using-opencv-817779c86a83

import cv2
import numpy as np

from simple_cb import simplest_cb

# Algorithm options
SIFT = "SIFT"
SURF = "SURF"
ORB = "ORB"
BRISK = "BRISK"

# Color options
GRAY = "GRAY"
COLORED = "COLORED"

# Overlay options
OVERLAY_MASK = "mask"
OVERLAY_WEIGHTED = "weighted"

# Configs
RATIO_DISTANCE = 0.67
MIN_MATCHES = 50
EQUALIZE_HISTOGRAM = False

SHOW_IMG_STEPS = False
SHOW_IMG_STEPS_DELAY = 10000

MATCH_ALGORITHM = SURF
OVERLAY_TYPE = OVERLAY_WEIGHTED
COLOR_TYPE = COLORED


class Image:
    UID = 0

    def __init__(self, d):
        if isinstance(d, str):
            if COLOR_TYPE == COLORED:
                self.img = cv2.imread(d, cv2.IMREAD_COLOR)
                self.img = simplest_cb(self.img, 1)
            else:
                self.img = cv2.imread(d, cv2.IMREAD_GRAYSCALE)
        else:
            self.img = d
        self.loaded = isinstance(d, str)
        self.uid = Image.UID
        Image.UID += 1

    def __eq__(self, other):
        if isinstance(other, Image):
            return other.uid == self.uid
        else:
            return super(self).__eq__(other)


def equalize_bgr(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


def insert_border(imgs):
    base_img = imgs[0]
    height, width = base_img.img.shape[:2]
    for img in imgs:
        img.img = cv2.copyMakeBorder(img.img, height, height, width, width, cv2.BORDER_CONSTANT)


def remove_border(img):
    if COLOR_TYPE == COLORED:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    _, thresh = cv2.threshold(gray, 3, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])

    new_img = [
        layer[y:(y + h), x:(x + w)] for layer in cv2.split(img)
    ]

    return cv2.merge(new_img)


def load_images(*args):
    return [Image(img) for img in args]


def create_kp_describer_matcher(t):
    assert t == SURF or t == SIFT or t == ORB or t == BRISK, "t param is invalid"

    if SURF == t or SIFT == t:
        if SURF == t:
            kp_describer = cv2.xfeatures2d.SURF_create()
        else:
            kp_describer = cv2.xfeatures2d.SIFT_create()
        index_params = {'algorithm': 0, 'trees': 5}
        search_params = {'checks': 1000}
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        return kp_describer, matcher

    elif ORB == t or BRISK == t:
        if ORB == t:
            kp_describer = cv2.ORB_create()
        else:
            kp_describer = cv2.BRISK_create()
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        return kp_describer, matcher


def match_images(img1, img2, match_algorithm):
    kp_describer, matcher = create_kp_describer_matcher(match_algorithm)

    if EQUALIZE_HISTOGRAM:
        img1 = equalize_bgr(img1)
        img2 = equalize_bgr(img2)

    kp1, des1 = kp_describer.detectAndCompute(img1, None)
    kp2, des2 = kp_describer.detectAndCompute(img2, None)

    raw_matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = []
    for m1, m2 in raw_matches:
        if m1.distance < RATIO_DISTANCE * m2.distance:
            good_matches.append(m1)

    return kp1, kp2, good_matches


def get_best_match(images, match_algorithm):
    print(f'Processing best match in {len(images)} images')

    # Not implemented. Gets the 2 first images
    img1, img2 = images[0], images[1]

    kp1, kp2, matchs = match_images(img1.img, img2.img, match_algorithm)

    return img1, img2, kp1, kp2, matchs

    # Below is a failed try to identify the best match pair
    # best_image = {
    #     'image_a': images[0],
    #     'image_b': images[1],
    #     'kp1': None,
    #     'kp2': None,
    #     'good_matches': []
    # }
    # for image_a in images:
    #     for image_b in images:
    #         if image_a != image_b:
    #             kp1, kp2, good_matches = match_images(image_a.img, image_b.img, match_algorithm)
    #             if len(good_matches) > len(best_image['good_matches']):
    #                 best_image['good_matches'] = good_matches
    #                 best_image['image_a'] = image_a
    #                 best_image['image_b'] = image_b
    #                 best_image['kp1'] = kp1
    #                 best_image['kp2'] = kp2
    #
    # return best_image['image_a'], best_image['image_b'], best_image['kp1'], best_image['kp2'], best_image['good_matches']


def homograph(img1, kp1, img2, kp2, good_matches, min_matches, show_matching, idx):
    print('Calculating homograph')
    if show_matching:
        good_matches_to_draw = [[p] for p in good_matches]
        match_image = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches_to_draw, None, flags=2)
        cv2.imshow(f'matching {idx}', match_image)
        cv2.waitKey(SHOW_IMG_STEPS_DELAY)

    if len(good_matches) >= min_matches:
        img1_key_points = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        img2_key_points = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        return cv2.findHomography(img1_key_points, img2_key_points, cv2.RANSAC, 5.0)
    return None, None


def panorama(imgs):
    imgs = load_images(*imgs)
    insert_border(imgs)

    panorama_img = None
    non_processed = [*imgs]

    i = 0
    while len(imgs) > 1:
        print(f'{len(imgs)} images left')
        i += 1

        image1, image2, kp1, kp2, good_matches = get_best_match(imgs, MATCH_ALGORITHM)
        imgs.remove(image1)
        imgs.remove(image2)

        img1 = image1.img
        img2 = image2.img

        # kp1, kp2, good_matches = match_images(img1, img2, MATCH_ALGORITHM)
        h1_2, _ = homograph(img1, kp1, img2, kp2, good_matches, MIN_MATCHES, SHOW_IMG_STEPS, i)
        if h1_2 is None:
            break

        height, width = img1.shape[:2]

        # warp perspective using the homograph of match
        img1_2 = cv2.warpPerspective(img1, h1_2, (width, height))

        # overlay images
        panorama_img = overlay_images(img1_2, img2)

        if SHOW_IMG_STEPS:
            cv2.imshow(f'panorama {i}', panorama_img)
            cv2.waitKey(SHOW_IMG_STEPS_DELAY)

        if image1.loaded:
            non_processed.remove(image1)
        if image2.loaded:
            non_processed.remove(image2)
        imgs.append(Image(panorama_img))

    panorama_img = remove_border(panorama_img)
    for non_processed_img in non_processed:
        non_processed_img.img = remove_border(non_processed_img.img)

    return panorama_img, [img.img for img in non_processed]


def overlay_images(img1_2, img2):
    if OVERLAY_TYPE == OVERLAY_WEIGHTED:
        panorama_img = cv2.addWeighted(img1_2, 0.5, img2, 0.5, 0)
    else:
        _, gray_1 = cv2.threshold(
            src=cv2.cvtColor(img1_2, cv2.COLOR_RGB2GRAY),
            thresh=1,
            maxval=255,
            type=cv2.THRESH_BINARY
        )
        _, gray_2 = cv2.threshold(
            src=cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY),
            thresh=1,
            maxval=255,
            type=cv2.THRESH_BINARY
        )

        mask = cv2.bitwise_xor(gray_1, gray_2)

        split_1 = cv2.split(img1_2)
        split_2 = cv2.split(img2)
        panorama_img = cv2.merge([
            cv2.add(
                cv2.bitwise_and(split_1[l_idx], mask),
                split_2[l_idx]
            )
            for l_idx in range(len(split_1))
        ])
    return panorama_img


def main():
    images = [
        './resources/img1.png',
        './resources/img2.png',
        './resources/img3.png',
    ]
    # images = [
    #     './resources/quarto1.jpg',
    #     './resources/quarto2.jpg',
    #     './resources/quarto3.jpg',
    # ]
    panorama_img, non_processed = panorama(images)

    for i in range(len(non_processed)):
        cv2.imshow(f'Error n{i + 1}', non_processed[i])

    if panorama_img is not None:
        cv2.imshow('Final panorama', panorama_img)
    else:
        error_img = np.zeros((640, 480, 1), np.uint8)
        cv2.imshow('Final panorama', error_img)

    cv2.waitKey(0)


if __name__ == '__main__':
    main()
