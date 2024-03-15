import sys
import time

import cv2
import numpy as np
import requests
from shapely.geometry import Polygon
from skimage.metrics import structural_similarity as compare_ssim

from common import fetch_image
from object_detection import ClassifierModel


def calc_roi_intersection(poly0, poly1):
    if len(poly1) > 2 and len(poly0) > 2:
        contour0 = np.array(poly0, dtype=np.float32)
        contour1 = np.array(poly1, dtype=np.float32)
        overlap_area = contour0.intersection(contour1).area
        return overlap_area
    return 0


class OverlapDetectContext:
    def __init__(self, name, task_id):
        self.name = name
        self.task_id = task_id
        self.params = None
        self.include_areas = None
        self.exclude_areas = None
        self.resize = None
        self.prev_frame = None
        self.prev_time = None
        self.rects = []
        self.prev_img_path = None
        for i in range(16):
            self.rects.append((0, 0, 0, 0))

    def proc_param(self, params):
        self.params = params
        self.resize = self.params.get('resize', [720, 480])
        roi = params.get('roi', {})
        include_areas = roi.get('includeAreas', [])
        exclude_areas = roi.get('excludeAreas', [])
        self.include_areas = [
            np.array([[int(x * self.resize[0]), int(y * self.resize[1])] for x, y in area], dtype=np.int32) for area in
            include_areas]
        self.exclude_areas = [
            np.array([[int(x * self.resize[0]), int(y * self.resize[1])] for x, y in area], dtype=np.int32) for area in
            exclude_areas]

    def check_prev_frame(self):
        if self.prev_frame is None:
            return False
        else:
            height, width = self.prev_frame.shape[:2]
            if height != self.resize[1] or width != self.resize[0]:
                self.prev_frame = None
                self.prev_time = None
                return False
            return True

    def process_contours(self, message, img, img_path):
        params = message.get("params", {})
        self.proc_param(params)
        frame = cv2.resize(img, (self.resize[0], self.resize[1]))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0)
        if not self.check_prev_frame():
            self.prev_img_path = img_path
            self.prev_frame = gray_frame
            self.prev_time = time.time()
            return []
        contour_results = self.calc_overlap_contours_v1(gray_frame, img_path)
        return contour_results


    def calc_overlap_contours(self, gray_frame):
        current_time = time.time()
        elapsed_time = current_time - self.prev_time
        contour_point = []
        (score, diff) = compare_ssim(self.prev_frame, gray_frame, win_size=91, full=True)
        diff = (diff * 255).astype("uint8")
        print("SSIM:{}".format(score))
        thresh = cv2.threshold(diff, 120, 255, cv2.THRESH_BINARY_INV)[1]
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_results = []
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            if w < 7 or h < 7:
                continue
            contour_point.append((x, y, w, h))
            for j in range(len(contour_point)):
                cx = contour_point[j][0]
                cy = contour_point[j][1]
                cw = contour_point[j][2]
                ch = contour_point[j][3]

                px = self.rects[j][0]
                py = self.rects[j][1]
                pw = self.rects[j][2]
                ph = self.rects[j][3]

                deltax = abs((cx + cw / 2) - (px + pw / 2))
                deltay = abs((cy + ch / 2) - (py + ph / 2))
                overlap_percent = 0.0
                if deltax < 20 and deltay < 20:
                    print("出现障碍物")

                    print("elapsed time is ", elapsed_time)
                    if elapsed_time > 2:

                        crop_img = gray_frame[cy:cy + ch, cx:cx + cw]
                        prev_crop_img = self.prev_frame[cy:cy + ch, cx:cx + cw]
                        (crop_score, crop_diff) = compare_ssim(crop_img, prev_crop_img, full=True)
                        crop_diff = (crop_diff * 255).astype("uint8")
                        print("crop_SSIM:{}".format(crop_score))
                        crop_thresh = cv2.threshold(crop_diff, 120, 255, cv2.THRESH_BINARY_INV)[1]
                        crop_contours, crop_hierarchy = cv2.findContours(crop_thresh.copy(), cv2.RETR_EXTERNAL,
                                                                         cv2.CHAIN_APPROX_SIMPLE)
                        for c1 in crop_contours:
                            if len(c1) >= 3:  # 确保至少有三个点

                                contour_points1 = [(pt[0][0] + cx, pt[0][1] + cy) for pt in c1]  # 提取点
                                contour_points1.append(contour_points1[0])  # 添加第一个点到末尾以闭合多边形
                                contour_shape = Polygon(contour_points1)  # 创建多边形
                                percents = []
                                for roi_poly in self.include_areas:
                                    poly = Polygon(roi_poly)
                                    if poly.area > 0:
                                        intersection_area = poly.intersection(contour_shape).area
                                        percent = intersection_area / poly.area
                                        if percent > 0:
                                            print('contour overlap percent:{}'.format(percent))
                                        percents.append(percent)
                                overlap_percent = max(percent for percent in percents)

                    else:
                        self.rects[j] = (cx, cy, cw, ch)
                else:
                    self.rects[j] = (cx, cy, cw, ch)
            contour_results.append({
                "id": int(j),
                "bBox": {
                    "x": float(cx / self.resize[0]),
                    "y": float(cy / self.resize[1]),
                    "w": float(cw / self.resize[0]),
                    "h": float(ch / self.resize[1]),
                },
                "overlapPercent": overlap_percent,
                "matched": False,
            })
        max_percent = max(
            (contour["overlapPercent"] if contour["overlapPercent"] is not None else 0 for contour in contour_results),
            default=0)
        if len(contour_results) == 0 or max_percent == 0 or elapsed_time >= 15:
            self.prev_frame = gray_frame
            self.prev_time = current_time
        return contour_results

    def calc_overlap_contours_v1(self, gray_frame, img_path):
        current_time = time.time()
        elapsed_time = current_time - self.prev_time
        contour_point = []
        (score, diff) = compare_ssim(self.prev_frame, gray_frame, win_size=91, full=True)
        diff = (diff * 255).astype("uint8")
        print("SSIM:{}".format(score))
        thresh = cv2.threshold(diff, 120, 255, cv2.THRESH_BINARY_INV)[1]
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_results = []
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            if w < 7 or h < 7:
                continue
            contour_point.append((x, y, w, h))
            for j in range(len(contour_point)):
                cx = contour_point[j][0]
                cy = contour_point[j][1]
                cw = contour_point[j][2]
                ch = contour_point[j][3]

                px = self.rects[j][0]
                py = self.rects[j][1]
                pw = self.rects[j][2]
                ph = self.rects[j][3]

                deltax = abs((cx + cw / 2) - (px + pw / 2))
                deltay = abs((cy + ch / 2) - (py + ph / 2))
                overlap_percent = 0.0
                if deltax < 20 and deltay < 20:
                    print("出现障碍物")

                    if len(c) >= 3:  # 确保至少有三个点
                        points = [pt[0] for pt in c]  # 提取点
                        points.append(points[0])  # 添加第一个点到末尾以闭合多边形
                        shape = Polygon(points)  # 创建多边形
                        if shape.is_valid:
                            percents = []
                            for roi_poly in self.include_areas:
                                poly = Polygon(roi_poly)
                                if poly.area > 0:
                                    intersection_area = poly.intersection(shape).area
                                    percent = intersection_area / poly.area
                                    if percent > 0:
                                        print('contour overlap percent:{}'.format(percent))
                                    percents.append(percent)
                            max_percent = max(percent for percent in percents)
                            contour_results.append({
                                "id": int(j),
                                "bBox": {
                                    "x": float(x / self.resize[0]),
                                    "y": float(y / self.resize[1]),
                                    "w": float(w / self.resize[0]),
                                    "h": float(h / self.resize[1]),
                                },
                                "overlapPercent": max_percent,
                                "matched": False,
                            })
                else:
                    self.rects[j] = (cx, cy, cw, ch)

        if len(contour_results) == 0 or elapsed_time >= 15:
            self.prev_frame = gray_frame
            self.prev_time = current_time
            self.prev_img_path = img_path
        return contour_results


class OccupancyDetector(ClassifierModel):
    def __init__(self, alg_name, args):
        super().__init__(alg_name, args)
        self.channel_ctx_dict = {}

    def get_channel_ctx(self, task_id):
        ctx = self.channel_ctx_dict.get(task_id)
        if ctx is None:
            ctx = OverlapDetectContext(self.name, task_id)
            self.channel_ctx_dict[task_id] = ctx
        return ctx

    def process(self, message):
        params = message.get("params", {})
        task_id = message['taskId']
        img_path = message['imgPath']
        img_url = self.args.addr + "/storage/" + message['bucket'] + '/' + message['imgPath']
        img = fetch_image(img_url)
        if img is None:
            print("image cannot be downloaded:{}", img_url)
            return
        ctx = self.get_channel_ctx(task_id)
        alg_result = self.process_image(message, img)
        alg_result["classResults"] = []
        contour_results = ctx.process_contours(message, img, img_path)
        alg_result["contourResults"] = contour_results
        threshold = params.get('overlapThreshold', 0.66)
        for contour in contour_results:
            # 检查每个元素的 overlapPercent 是否大于 0.5
            if contour["overlapPercent"] >= threshold:
                # 如果是，设置 alarm 字段为 True
                contour["v"] = True
            else:
                # 可选：如果你需要确保所有对象都有 alarm 字段，可以设置为 False
                # 如果只需要标记大于0.5的，这行可以省略
                contour["matched"] = False
        maxOverlapPercent = max(
            (contour["overlapPercent"] if contour["overlapPercent"] is not None else 0 for contour in contour_results),
            default=0)
        alg_result["maxOverlapPercent"] = maxOverlapPercent
        print("max overlap percent:{}".format(maxOverlapPercent))
        if maxOverlapPercent >= threshold:
            alg_result["alarmFlag"] = True
            alg_result["prevImgPath"] = ctx.prev_img_path

        # 这里可以添加图像处理逻辑
        # alg_result = self.process_image(message, img)  # 假设这是你的图像处理函数
        if alg_result is not None and alg_result["maxOverlapPercent"] is not None:
            # 上传结果
            target_url = '{}/api/iss/alarm/task/{}/job/{}/result'.format(self.args.addr, task_id, self.name)
            try:
                response = requests.post(target_url, json=alg_result)
                if response.status_code == 200:
                    print("上传成功")
                else:
                    print("上传失败，状态码:", response.status_code)
                sys.stdout.flush()
            except Exception as e:
                print("上传出错:", str(e))
                sys.stdout.flush()
