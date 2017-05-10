import numpy as np
import cv2
from skimage.transform import resize
from scipy.ndimage.measurements import label
from vehicle_features import FeatureExtractor

class VehicleTracker(object):

    def __init__(self, scaler, classifier, image_shape):
        self.scaler = scaler
        self.classifier = classifier
        self.image_shape = image_shape
        self.detections_history = []

    def process(self, frame, draw_detections=True, draw_search_rectangles=False):
        self.find_cars(frame, draw_search_rectangles=draw_search_rectangles)
        if draw_detections is False:
            return frame
        for coordinate in self.detections(detections_history_count=20):
            (x1, y1, x2, y2) = coordinate
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
        return frame

    def find_cars(self, image, draw_search_rectangles):
        scales = np.array([.3, .5, .65, .8])
        y_tops = np.array([.6, .57, .56, .55])
        window_size = 64
        frame_detections = np.empty([0, 4], dtype=np.int64)
        for scale, y_top in zip(scales, y_tops):
            scale_detections = self.detections_for_scale(image, scale, y_top, window_size) 
            frame_detections = np.append(frame_detections, scale_detections, axis=0)
            if draw_search_rectangles is True:
                x_range = np.linspace(
                    0,
                    int(image.shape[1]),
                    4
                ).astype(np.int)
                for x1 in x_range:
                    x2 = int(x1 + window_size // scale)
                    y1 = int(y_top * image.shape[0])
                    y2 = int(y_top * image.shape[0] + window_size // scale)
                    cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 255, 255), thickness=3)
        detections, self.heatmap = self.merge_detections(frame_detections, threshold=1)
        self.detections_history.append(detections)

    def detections(self, detections_history_count):
        history = self.detections_history[-detections_history_count:]
        detections, _ = self.merge_detections(
            np.concatenate(np.array(history)),
            threshold=min(len(history), 15)
        )
        return detections

    def merge_detections(self, detections, threshold):
        heatmap = np.zeros((self.image_shape[0], self.image_shape[1])).astype(np.float)
        # Add heat to each box in box list
        heatmap = self.add_heat(heatmap, detections)
        # Apply threshold to help remove false positives
        heatmap[heatmap < threshold] = 0
        heatmap = np.clip(heatmap, 0, 255)
        labels = label(heatmap)
        merged_detections = np.empty([0, 4], dtype=np.int64)
        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            merged_detections = np.append(
                merged_detections,
                [[np.min(nonzerox), np.min(nonzeroy), np.max(nonzerox), np.max(nonzeroy)]],
                axis=0
            )
        # Return the image
        return (merged_detections, heatmap)

    def detections_for_scale(self, image, image_scale, y_top, window_size):
        (height, width, num_channels) = image.shape
        scaled_image = resize(
            (image / 255.).astype(np.float64), 
            (int(height * image_scale), int(width * image_scale), num_channels), 
            preserve_range=True
        ).astype(np.float32)
        extractor = FeatureExtractor(scaled_image)

        (scaled_height, scaled_width, _) = scaled_image.shape
        scaled_y_top = int(scaled_height * y_top)
        x_range = np.linspace(
            0,
            scaled_width - window_size,
            (scaled_width + window_size // 3) // (window_size // 3)
        ).astype(np.int)
        detections = np.empty([0, 4], dtype=np.int)
        for scaled_x in x_range:
            features = extractor.feature_vector(scaled_x, scaled_y_top, window_size)
            features = self.scaler.transform(np.array(features).reshape(1, -1))
            if self.classifier.predict(features)[0] == 1:
                detections = np.append(
                    detections, 
                    [[scaled_x, scaled_y_top, scaled_x + window_size, scaled_y_top + window_size]], 
                    axis=0
                )
        return (detections / image_scale).astype(np.int)

    def add_heat(self, heatmap, coordinates):
        for coordinate in coordinates:
            (x1, y1, x2, y2) = coordinate
            heatmap[y1:y2, x1:x2] += 1
        return heatmap