import math


class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 1
        # 以往紀錄
        self.history = {}

    def search_history(self, cx, cy):
        ans = 0
        for id, pt in self.history.items():
            dist = math.hypot(cx - pt[0], cy - pt[1])
            if dist < 100:
                ans = id
                break
        if ans is not 0:
            self.history.pop(ans)
        return ans

    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 100:
                    self.center_points[id] = (cx, cy, 0)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                past_id = self.search_history(cx, cy)
                if past_id == 0:
                    self.center_points[self.id_count] = (cx, cy, 0)
                    objects_bbs_ids.append([x, y, w, h, self.id_count])
                    self.id_count += 1
                else:
                    self.center_points[past_id] = (cx, cy, 0)
                    objects_bbs_ids.append([x, y, w, h, past_id])



        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.history.update(self.center_points)
        self.center_points = new_center_points.copy()
        Del = []
        for id, pt in self.history.items():
            if pt[2] > 5:  # 保留5偵
                Del.append(id)
            else:
                self.history[id] = (pt[0], pt[1], pt[2] + 1)
        for i in Del:
            self.history.pop(i)

        #if len(self.center_points) > 0:
            #print(self.center_points)

        return objects_bbs_ids



