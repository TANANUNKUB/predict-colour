import onnxruntime
import numpy as np
import cv2

class PREDICT_COLOUR:

    def __init__(self, onnx_file):
        self.classes =  [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush"
        ]

        self.ort_session, self.input_names, self.output_names = self.load_model(onnx_file)
        self.image_height=0
        self.image_width=0
        self.input_height=640
        self.input_width=640

    def load_model(self, model_name):

        opt_session = onnxruntime.SessionOptions()
        opt_session.enable_mem_pattern = False
        opt_session.enable_cpu_mem_arena = False
        opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

        ort_session = onnxruntime.InferenceSession(model_name)
        model_inputs = ort_session.get_inputs()
        model_output = ort_session.get_outputs()
        input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        output_names = [model_output[i].name for i in range(len(model_output))]
        return ort_session, input_names, output_names

    def predict(self, input_tensor, conf_thresold=0.4):

        outputs = self.ort_session.run(self.output_names, {self.input_names[0]: input_tensor})[0]
        predictions = np.squeeze(outputs).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > conf_thresold, :]
        scores = scores[scores > conf_thresold]  

        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = predictions[:, :4]

        #rescale box
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.image_width, self.image_height, self.image_width, self.image_height])
        boxes = boxes.astype(np.int32)

        indices = self.nms(boxes, scores, 0.5)

        return boxes, indices, scores, class_ids

    def nms(self, boxes, scores, iou_threshold):
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]

        keep_boxes = []
        while sorted_indices.size > 0:
            # Pick the last box
            box_id = sorted_indices[0]
            keep_boxes.append(box_id)

            # Compute IoU of the picked box with the rest
            ious = self.compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

            # Remove boxes with IoU over the threshold
            keep_indices = np.where(ious < iou_threshold)[0]

            # print(keep_indices.shape, sorted_indices.shape)
            sorted_indices = sorted_indices[keep_indices + 1]

        return keep_boxes

    def compute_iou(self, box, boxes):
        box = self.xywh2xyxy(box)
        boxes = self.xywh2xyxy(boxes)
        # Compute xmin, ymin, xmax, ymax for both boxes
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])
        # Compute intersection area
        intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
        # Compute union area
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - intersection_area
        # Compute IoU
        iou = intersection_area / union_area

        return iou

    def xywh2xyxy(self, x):
        # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2

        return y

    def preview(self, image, boxes, indices, scores, class_ids):
        image_draw = image.copy()
        for (bbox, score, label) in zip(self.xywh2xyxy(boxes[indices]), scores[indices], class_ids[indices]):
            bbox = bbox.round().astype(np.int32).tolist()
            cls_id = int(label)
            cls = self.classes[cls_id]
            color = (0,255,0)
            #predict colour
            x1, y1, x2, y2 = bbox
            if x1 < 0 :x1 = 0
            if y1 < 0 :y1 = 0
            if x2 < 0 :x2 = 0
            if y2 < 0 :y2 = 0

            crop_img = image[y1:y2, x1:x2]
            
            pred_colour = self.pred_colour(crop_img)
            cv2.rectangle(image_draw, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
            #predict text
            cv2.putText(image_draw,
                        f'{cls}:{int(score*100)}%', (bbox[0], bbox[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, [225, 0, 0],
                        thickness=2)
            #colour text
            cv2.putText(image_draw,
                        pred_colour, (bbox[0], bbox[1]+30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, [0, 0, 255],
                        thickness=2)
        
        return image_draw
    def rgb_to_hsv(self, rgb_value):
        r, g, b = rgb_value
        r /= 255.0
        g /= 255.0
        b /= 255.0
        max_value = max(r, g, b)
        min_value = min(r, g, b)
        delta = max_value - min_value
        # Calculate Hue
        if delta == 0:
            h = 0  # undefined, but we can set it to 0
        elif max_value == r:
            h = ((g - b) / delta) % 6
        elif max_value == g:
            h = ((b - r) / delta) + 2
        else:
            h = ((r - g) / delta) + 4
        h *= 60  # Convert to degrees
        if h < 0:
            h += 360
        # Calculate Saturation
        if max_value == 0:
            s = 0
        else:
            s = delta / max_value
        # Calculate Value
        v = max_value
        return h, s, v

    def pred_colour(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_value = cv2.mean(image_rgb)[:3]
        h, s, v = self.rgb_to_hsv(rgb_value)         
        if s < 0.03 and v > 0.95 :colour = 'white'
        elif v < 0.15:colour = 'black'
        elif h > 345 :colour = 'red'
        elif h > 300 :colour = 'pink'
        elif h > 255 :colour = 'purple'
        elif h > 220 :colour = 'blue'
        elif h > 160 :colour = 'sky blue'
        elif h > 80 :colour = 'green'
        elif h > 40 :colour = 'yellow'
        elif h > 15 :colour = 'orange'
        elif h < 15 :colour = 'red' 
        return colour

    def __call__(self,image_path, output_path):
        image = cv2.imread(image_path)
        self.image_height, self.image_width = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (self.input_width, self.input_height))
        input_image = resized / 255.0
        input_image = input_image.transpose(2,0,1)
        input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)
        boxes, indices, scores, class_ids = self.predict(input_tensor)
        image_preview = self.preview(image, boxes, indices, scores, class_ids)
        cv2.imwrite(output_path, image_preview)
        return output_path