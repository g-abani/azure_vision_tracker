
import numpy as np
from scipy.spatial import distance

class CentroidTracker:
    def __init__(self, max_distance=50):
        self.next_object_id = 0
        self.objects = {}  # object_id -> centroid
        self.labels = {}   # object_id -> label
        self.max_distance = max_distance

    def update(self, detections):
        """
        detections: list of dicts with keys 'label' and 'bbox' (x, y, w, h)
        Returns: dict of object_id -> updated bounding box and label
        """
        if not detections:
            return {}

        input_centroids = []
        input_labels = []
        bboxes = []

        for det in detections:
            x, y, w, h = det["x"], det["y"], det["w"], det["h"]
            cx, cy = x + w // 2, y + h // 2
            input_centroids.append((cx, cy))
            input_labels.append(det["label"])
            bboxes.append((x, y, w, h))

        if not self.objects:
            for i, (centroid, label) in enumerate(zip(input_centroids, input_labels)):
                self.objects[self.next_object_id] = centroid
                self.labels[self.next_object_id] = label
                self.next_object_id += 1
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = distance.cdist(np.array(object_centroids), np.array(input_centroids))
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if D[row, col] < self.max_distance:
                    object_id = object_ids[row]
                    self.objects[object_id] = input_centroids[col]
                    self.labels[object_id] = input_labels[col]
                    used_rows.add(row)
                    used_cols.add(col)

            # Add unmatched new detections
            for i in range(len(input_centroids)):
                if i not in used_cols:
                    self.objects[self.next_object_id] = input_centroids[i]
                    self.labels[self.next_object_id] = input_labels[i]
                    self.next_object_id += 1

        tracked = {}
        for object_id, centroid in self.objects.items():
            label = self.labels[object_id]
            for i, c in enumerate(input_centroids):
                if centroid == c:
                    x, y, w, h = bboxes[i]
                    tracked[object_id] = {
                        "label": label,
                        "bbox": (x, y, w, h),
                        "centroid": centroid
                    }
        return tracked
