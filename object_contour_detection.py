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


def calculate_result_similarity(item1, item2):
    # 定义权重
    weights = {'position': 0.4, 'size': 0.4, 'overlapPercent': 0.2}

    # 计算位置相似性
    position_diff = ((item1['bBox']['x'] - item2['bBox']['x']) ** 2 +
                     (item1['bBox']['y'] - item2['bBox']['y']) ** 2) ** 0.5
    position_similarity = max(0, 1 - position_diff)  # 假设差异为0时完全相同，随差异增大逐渐减少到0

    # 计算大小相似性
    size_diff = ((item1['bBox']['w'] - item2['bBox']['w']) ** 2 +
                 (item1['bBox']['h'] - item2['bBox']['h']) ** 2) ** 0.5
    size_similarity = max(0, 1 - size_diff)

    # 计算重叠百分比相似性
    overlap_diff = abs(item1['overlapPercent'] - item2['overlapPercent'])
    overlap_similarity = max(0, 1 - overlap_diff)

    # 综合评分
    total_similarity = (position_similarity * weights['position'] +
                        size_similarity * weights['size'] +
                        overlap_similarity * weights['overlapPercent'])

    return total_similarity


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
        self.alarm_time = None
        self.prev_contour_results = []
        self.rects = []
        self.prev_img_path = None
        for i in range(32):
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

    def get_elapsed_time(self):
        return time.time() - self.prev_time

    def get_alarm_duration(self):
        if self.alarm_time is None:
            return 0
        return time.time() - self.alarm_time

    def process_contours(self, message, img, img_path):
        params = message.get("params", {})
        alarm_interval = params.get('alarmInterval', 60)
        self.proc_param(params)
        frame = cv2.resize(img, (self.resize[0], self.resize[1]))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0)
        if not self.check_prev_frame():
            self.prev_img_path = img_path
            self.prev_frame = gray_frame
            self.prev_time = time.time()
            return []
        contour_results = self.calc_overlap_contours_v2(gray_frame)
        max_alarm_duration = max(alarm_interval / 2, 15)
        if self.get_elapsed_time() >= 15 or self.get_alarm_duration() >= max_alarm_duration:
            current_time = time.time()
            self.prev_frame = gray_frame
            self.prev_time = current_time
            self.prev_img_path = img_path
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
            if w < 20 or h < 20:
                continue
            contour_point.append((x, y, w, h))

        for j in range(len(contour_point)):
            if j > len(self.rects):
                break
            cx = contour_point[j][0]
            cy = contour_point[j][1]
            cw = contour_point[j][2]
            ch = contour_point[j][3]

            px = self.rects[j][0]
            py = self.rects[j][1]
            pw = self.rects[j][2]
            ph = self.rects[j][3]
            intersection_area0 = 0

            deltax = abs((cx + cw / 2) - (px + pw / 2))
            deltay = abs((cy + ch / 2) - (py + ph / 2))
            points0 = [pt[0] for pt in c]  # 提取点
            shape0 = Polygon(points0)  # 创建多边形
            for roi_poly in self.include_areas:
                poly = Polygon(roi_poly)
                if poly.area > 0:
                    intersection_area0 += poly.intersection(shape0).area
            if deltax < 1 and deltay < 1:
                print('delta < 1')
            else:
                self.rects[j] = (cx, cy, cw, ch)

            if intersection_area0 > 10:
                print("出现障碍物")
                crop_img = gray_frame[cy:cy + ch, cx:cx + cw]
                prev_crop_img = self.prev_frame[cy:cy + ch, cx:cx + cw]
                (crop_score, crop_diff) = compare_ssim(prev_crop_img, crop_img, full=True)
                crop_diff = (crop_diff * 255).astype("uint8")
                crop_thresh = cv2.threshold(crop_diff, 120, 255, cv2.THRESH_BINARY_INV)[1]
                crop_contours, crop_hierarchy = cv2.findContours(crop_thresh.copy(), cv2.RETR_EXTERNAL,
                                                                 cv2.CHAIN_APPROX_SIMPLE)
                print(f"crop_SSIM:{crop_score}")
                for cc in crop_contours:
                    if cv2.contourArea(cc) < 10 or len(cc) <= 3:
                        continue
                    points = [(pt[0][0] + cx, pt[0][1] + cy) for pt in cc]  # 提取点
                    (ccx, ccy, ccw, cch) = cv2.boundingRect(cc)
                    ccx += cx
                    ccy += cy
                    points.append(points[0])  # 添加第一个点到末尾以闭合多边形
                    shape = Polygon(points)  # 创建多边形
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
                    if max_percent > 0:
                        contour_results.append({
                            "id": int(j),
                            "bBox": {
                                "x": float(ccx / self.resize[0]),
                                "y": float(ccy / self.resize[1]),
                                "w": float(ccw / self.resize[0]),
                                "h": float(cch / self.resize[1]),
                            },
                            "overlapPercent": max_percent,
                            "matched": False,
                        })
                        self.prev_time = current_time
            else:
                self.rects[j] = (cx, cy, cw, ch)

        if len(contour_results) == 0 and self.get_elapsed_time() >= 15:
            self.prev_frame = gray_frame
            self.prev_time = current_time
            self.prev_img_path = img_path
        return contour_results

    def calc_overlap_contours_v2(self, gray_frame):
        current_time = time.time()
        (score, diff) = compare_ssim(self.prev_frame, gray_frame, win_size=91, full=True)
        diff = (diff * 255).astype("uint8")
        print("SSIM:{}".format(score))
        thresh = cv2.threshold(diff, 120, 255, cv2.THRESH_BINARY_INV)[1]
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_results = []
        for j, c in enumerate(contours):
            (cx, cy, cw, ch) = cv2.boundingRect(c)
            if cw < 20 or ch < 20:
                continue
            intersection_area0 = 0
            points0 = [pt[0] for pt in c]  # 提取点
            points0.append(points0[0])  # 添加第一个点到末尾以闭合多边形
            shape0 = Polygon(points0)  # 创建多边形
            if not shape0.is_valid:
                print('shape0 is not valid')
                shape0 = shape0.buffer(0)
                if not shape0.is_valid:
                    print('shape0 is still not valid, will be ignored')
                    continue
            for roi_poly in self.include_areas:
                poly = Polygon(roi_poly)
                if poly.area > 0:
                    intersection_area0 += poly.intersection(shape0).area

            if intersection_area0 > 10:
                print("出现障碍物")
                crop_img = gray_frame[cy:cy + ch, cx:cx + cw]
                prev_crop_img = self.prev_frame[cy:cy + ch, cx:cx + cw]
                (crop_score, crop_diff) = compare_ssim(prev_crop_img, crop_img, full=True)
                crop_diff = (crop_diff * 255).astype("uint8")
                crop_thresh = cv2.threshold(crop_diff, 120, 255, cv2.THRESH_BINARY_INV)[1]
                crop_contours, crop_hierarchy = cv2.findContours(crop_thresh.copy(), cv2.RETR_EXTERNAL,
                                                                 cv2.CHAIN_APPROX_SIMPLE)
                print(f"crop_SSIM:{crop_score}")
                for jj, cc in enumerate(crop_contours):
                    if cv2.contourArea(cc) < 10 or len(cc) <= 3:
                        continue
                    points = [(pt[0][0] + cx, pt[0][1] + cy) for pt in cc]  # 提取点
                    (ccx, ccy, ccw, cch) = cv2.boundingRect(cc)
                    ccx += cx
                    ccy += cy
                    points.append(points[0])  # 添加第一个点到末尾以闭合多边形
                    shape = Polygon(points)  # 创建多边形
                    if not shape0.is_valid:
                        print('shape is not valid')
                        shape = shape.buffer(0)
                        if not shape.is_valid:
                            print('shape is still not valid, will be ignored')
                            continue

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
                    if max_percent > 0:
                        contour_results.append({
                            "id": int(j),
                            "bBox": {
                                "x": float(ccx / self.resize[0]),
                                "y": float(ccy / self.resize[1]),
                                "w": float(ccw / self.resize[0]),
                                "h": float(cch / self.resize[1]),
                            },
                            "overlapPercent": max_percent,
                            "matched": False,
                        })
        results = []
        if 0 < len(contour_results) == len(self.prev_contour_results):
            for i, cr in enumerate(contour_results):
                pcr = self.prev_contour_results[i]
                score = calculate_result_similarity(cr, pcr)
                print(f"Similarity score: {score}")
                if score > 0.95:
                    results.append(cr)
        elif len(contour_results) > 0 == len(self.prev_contour_results):
            self.prev_time = current_time

        self.prev_contour_results = contour_results
        return results


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
        prev_img_url = ctx.prev_img_path
        if prev_img_url is None:
            ctx.process_contours(message, img, img_path)
            return
        contour_results = ctx.process_contours(message, img, img_path)
        alg_result["contourResults"] = contour_results
        threshold = params.get('overlapThreshold', 0.66)
        alarm_interval = params.get('alarmInterval', 120)
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
        print("max overlap percent:{} / threshold: {}".format(maxOverlapPercent, threshold))
        alarm_flag = False
        now = time.time()
        if maxOverlapPercent >= threshold:
            alarm_flag = True
            ctx.prev_time = now
            if ctx.alarm_time is None:
                ctx.alarm_time = now
            alg_result["alarmFlag"] = True
            alg_result["prevImgPath"] = prev_img_url
        else:
            ctx.alarm_time = None

        # 这里可以添加图像处理逻辑
        # alg_result = self.process_image(message, img)  # 假设这是你的图像处理函数
        if alg_result is not None and alg_result["maxOverlapPercent"] is not None:
            if True:  # not alarm_flag or ctx.alarm_time is None or now - ctx.alarm_time >= alarm_interval / 4:
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
