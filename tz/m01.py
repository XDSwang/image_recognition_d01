import cv2
import os

def r_int(s: str):
    """将字符串分割成整数元组"""
    i = s.split("_")
    return int(i[0]), int(i[1])


def opencv_match(img_d, template_d, threshold=0.3):
    sift = cv2.SIFT_create()
    img1 = cv2.imread(img_d, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(template_d, cv2.IMREAD_GRAYSCALE)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    print(matches)
    result = []
    for match in matches:
        m1 = match[0]  # Get the first match
        if m1.distance < threshold * match[1].distance:
            pt1 = kp1[m1.queryIdx].pt
            pt2 = kp2[m1.trainIdx].pt
            a = str(round(pt1[0]) - round(pt2[0]))
            b = str(round(pt1[1]) - round(pt2[1]))
            result.append(a + "_" + b)

    if len(result) > 0:  # Check if there are enough matches to draw a rectangle
        unique_results = []
        for id in result:
            if id not in unique_results:
                unique_results.append(id)
        return unique_results
    else:
        return None



def render_matched_boxes(image_path, template_path, matches):
    img = cv2.imread(image_path)
    template = cv2.imread(template_path)
    h, w = template.shape[:2]

    for s in matches:
        point = r_int(s)
        cv2.rectangle(img, (point[0], point[1]), (point[0] + w, point[1] + h), (0, 0, 255), 2)

    return img


if __name__ == "__main__":
    src = os.path.split(os.path.realpath(__file__))[0]
    # a = "/mmexport1693120102165.jpg"
    # b = "/m01.jpg"
    a="/screenshot.png"
    b="/searchadd0508.png"
    matches = opencv_match(src+a, src+b, 0.3)
    print(matches)
    if matches is not None:
        result_image = render_matched_boxes(src+a, src+b, matches)
        cv2.namedWindow("Matching Result", 0)
        cv2.imshow("Matching Result", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No matches found.")
