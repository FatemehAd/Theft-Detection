import os
import csv
from datetime import datetime
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import logging
import threading
import queue

# Disable Qt backend to avoid threading issues
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up CSV file for logging
CSV_FILE = "storeroom_events.csv"
def init_csv():
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Person ID", "Item", "Event", "Confidence", "Box (startX, startY, endX, endY)"])

# Construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2, help="minimum probability to filter weak predictions")
ap.add_argument("-r", "--resolution", type=str, default="640x480", help="video resolution (e.g., 640x480)")
ap.add_argument("-s", "--skip-frames", type=int, default=1, help="number of frames to skip between processing")
ap.add_argument("-d", "--distance", type=int, default=100, help="pixel distance for person-item association")
ap.add_argument("-t", "--timeout", type=int, default=300, help="frames before taken item is considered stale")
args = vars(ap.parse_args())

# Validate model files
if not os.path.isfile(args["prototxt"]):
    logger.error(f"Prototxt file not found: {args['prototxt']}")
    exit(1)
if not os.path.isfile(args["model"]):
    logger.error(f"Model file not found: {args['model']}")
    exit(1)

# Initialize class labels and colors
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", 
           "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", 
           "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load the pre-trained model
logger.info("Loading model...")
try:
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
except cv2.error as e:
    logger.error(f"Failed to load Caffe model: {e}")
    exit(1)

# Initialize video stream parameters
CAM_URL = f"http://192.168.232.154:4747/video/{args['resolution']}"
MAX_RECONNECT_ATTEMPTS = 5
INITIAL_RECONNECT_DELAY = 2.0
QUEUE_SIZE = 30

# Frame queue for multithreading
frame_queue = queue.Queue(maxsize=QUEUE_SIZE)

def non_max_suppression(boxes, scores, overlapThresh=0.3):
    """Apply Non-Maximum Suppression to filter overlapping boxes."""
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes, dtype="float")
    scores = np.array(scores)
    
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)[::-1]
    
    pick = []
    while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[1:]]
        
        idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > overlapThresh)[0] + 1)))
    
    return pick

def compute_iou(box1, box2):
    """Compute Intersection over Union (IoU) for two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_2, y2_2)
    
    inter_area = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)
    box1_area = (x2_1 - x1_1 + 1) * (y2_1 - y1_1 + 1)
    box2_area = (x2_2 - x1_2 + 1) * (y2_2 - y1_2 + 1)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def video_capture_thread(url, queue):
    """Thread to capture video frames and put them in a queue."""
    attempt = 0
    while attempt < MAX_RECONNECT_ATTEMPTS:
        vs = cv2.VideoCapture(url)
        vs.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        vs.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        if not vs.isOpened():
            logger.error(f"Failed to open stream at {url}. Attempt {attempt + 1}/{MAX_RECONNECT_ATTEMPTS}")
            time.sleep(INITIAL_RECONNECT_DELAY * (2 ** attempt))
            attempt += 1
            continue
        
        while True:
            ret, frame = vs.read()
            if not ret:
                logger.warning("Stream disconnected. Attempting to reconnect...")
                vs.release()
                time.sleep(INITIAL_RECONNECT_DELAY * (2 ** attempt))
                attempt += 1
                break
            try:
                if queue.full():
                    try:
                        queue.get_nowait()
                        logger.debug("Dropped oldest frame due to full queue")
                    except queue.Empty:
                        pass
                queue.put_nowait(frame)
            except queue.Full:
                logger.debug("Queue full, skipping frame")
                continue
        if attempt >= MAX_RECONNECT_ATTEMPTS:
            logger.error("Max reconnection attempts reached. Exiting...")
            break
    queue.put(None)

# Initialize CSV and tracking variables
init_csv()
fps = FPS().start()
tracked_objects = {}
taken_items = {}
person_id_counter = 0
person_ids = {}
frame_count = 0
bottle_confidence_history = {}
object_id_counter = {"bottle": 0, "person": 0}

# Start video capture thread
logger.info(f"Starting video stream from DroidCam at {CAM_URL}...")
capture_thread = threading.Thread(target=video_capture_thread, args=(CAM_URL, frame_queue), daemon=True)
capture_thread.start()
time.sleep(2.0)

def assign_object_id(cls, box, existing_objects):
    """Assign a persistent ID based on IoU and centroid proximity."""
    startX, startY, endX, endY = box
    centroid_x = (startX + endX) / 2
    centroid_y = (startY + endY) / 2
    
    best_iou = 0
    best_id = None
    for obj_id, data in existing_objects.items():
        if data["class"] == cls:
            px = data["box"]
            iou = compute_iou(box, px)
            if iou > 0.5:  # IoU threshold for matching
                data["box"] = box
                return obj_id
            # Fallback to centroid distance if IoU is low
            p_centroid_x = (px[0] + px[2]) / 2
            p_centroid_y = (px[1] + px[3]) / 2
            distance = np.sqrt((centroid_x - p_centroid_x)**2 + (centroid_y - p_centroid_y)**2)
            if distance < args["distance"] and iou > best_iou:
                best_iou = iou
                best_id = obj_id
    
    if best_id:
        existing_objects[best_id]["box"] = box
        return best_id
    
    global object_id_counter
    object_id_counter[cls] += 1
    return f"{cls}_{object_id_counter[cls]}"

def log_event(person_id, item, event, confidence, box):
    """Log event to console and CSV."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    box_str = f"({box[0]}, {box[1]}, {box[2]}, {box[3]})"
    logger.info(f"{timestamp} - {person_id} {event} {item} (Confidence: {confidence:.2f}, Box: {box_str})")
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, person_id, item, event, f"{confidence:.2f}", box_str])

# Main loop
while True:
    try:
        frame = frame_queue.get(timeout=5.0)
        if frame is None:
            logger.error("No more frames. Exiting...")
            break
        
        if frame_count % args["skip_frames"] != 0:
            frame_count += 1
            continue
        
        frame = imutils.resize(frame, width=400)
        (h, w) = frame.shape[:2]
        resized_image = cv2.resize(frame, (300, 300))
        blob = cv2.dnn.blobFromImage(resized_image, 1/127.5, (300, 300), 127.5, swapRB=True)
        net.setInput(blob)
        predictions = net.forward()

        current_detections = {}
        boxes, scores, classes = [], [], []
        for i in np.arange(0, predictions.shape[2]):
            confidence = predictions[0, 0, i, 2]
            if confidence > args["confidence"]:
                idx = int(predictions[0, 0, i, 1])
                box = predictions[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                boxes.append([startX, startY, endX, endY])
                scores.append(confidence)
                classes.append(idx)

        nms_indices = non_max_suppression(np.array(boxes), scores)
        for i in nms_indices:
            startX, startY, endX, endY = boxes[i]
            idx = classes[i]
            confidence = scores[i]
            
            if CLASSES[idx] in ["bottle", "person"]:
                cls = CLASSES[idx]
                obj_id = assign_object_id(cls, (startX, startY, endX, endY), 
                                        person_ids if cls == "person" else tracked_objects)
                
                if cls == "bottle":
                    bottle_confidence_history[obj_id] = bottle_confidence_history.get(obj_id, []) + [confidence]
                    bottle_confidence_history[obj_id] = bottle_confidence_history[obj_id][-5:]
                    avg_confidence = np.mean(bottle_confidence_history[obj_id])
                    if avg_confidence < args["confidence"]:
                        continue
                    confidence = avg_confidence
                
                current_detections[obj_id] = {
                    "class": cls,
                    "box": (startX, startY, endX, endY),
                    "confidence": confidence
                }
                label = f"{obj_id}: {confidence * 100:.2f}%"
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        # Clean up stale taken items
        for obj_id in list(taken_items.keys()):
            if frame_count - taken_items[obj_id]["frame"] > args["timeout"]:
                logger.info(f"Removed stale taken item {obj_id} (taken by {taken_items[obj_id]['person_id']})")
                del taken_items[obj_id]

        # Track objects and detect taken/returned events
        for obj_id, obj_data in list(tracked_objects.items()):
            if obj_id not in current_detections and obj_data["class"] == "bottle":
                if "person_id" in obj_data:
                    taken_items[obj_id] = {
                        "person_id": obj_data["person_id"],
                        "class": obj_data["class"],
                        "confidence": obj_data["confidence"],
                        "frame": frame_count,
                        "box": obj_data["box"]
                    }
                    log_event(obj_data["person_id"], obj_data["class"], "taken", 
                             obj_data["confidence"], obj_data["box"])
                    cv2.putText(frame, f"{obj_data['person_id']} took {obj_data['class']}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                del tracked_objects[obj_id]

        # Check for returned items
        for obj_id, obj_data in current_detections.items():
            if obj_data["class"] == "bottle":
                # Check for exact ID match in taken_items
                if obj_id in taken_items:
                    person_id = taken_items[obj_id]["person_id"]
                    log_event(person_id, obj_data["class"], "returned", obj_data["confidence"], obj_data["box"])
                    cv2.putText(frame, f"{person_id} returned {obj_data['class']}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    del taken_items[obj_id]
                else:
                    # Check for potential return by IoU with taken items
                    for taken_id, taken_data in list(taken_items.items()):
                        if taken_data["class"] == "bottle":
                            iou = compute_iou(obj_data["box"], taken_data["box"])
                            if iou > 0.5:
                                person_id = taken_data["person_id"]
                                log_event(person_id, obj_data["class"], "returned", obj_data["confidence"], obj_data["box"])
                                cv2.putText(frame, f"{person_id} returned {obj_data['class']}", 
                                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                del taken_items[taken_id]
                                break

        # Associate bottles with persons
        for obj_id, obj_data in current_detections.items():
            if obj_data["class"] == "bottle":
                for person_id, person_data in current_detections.items():
                    if person_data["class"] == "person":
                        p_startX, p_startY, _, _ = person_data["box"]
                        startX, startY, _, _ = obj_data["box"]
                        if abs(p_startX - startX) < args["distance"] and abs(p_startY - startY) < args["distance"]:
                            tracked_objects[obj_id] = {
                                "class": obj_data["class"],
                                "person_id": person_id,
                                "confidence": obj_data["confidence"],
                                "last_seen": frame_count,
                                "box": obj_data["box"]
                            }
                            break
                else:
                    tracked_objects[obj_id] = {
                        "class": obj_data["class"],
                        "confidence": obj_data["confidence"],
                        "last_seen": frame_count,
                        "box": obj_data["box"]
                    }

        cv2.imshow("Storeroom Tracker", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        fps.update()
        frame_count += 1

    except queue.Empty:
        logger.warning("No frame received within timeout. Checking connection...")
        continue

# Cleanup
fps.stop()
logger.info(f"Elapsed Time: {fps.elapsed():.2f}s")
logger.info(f"Approximate FPS: {fps.fps():.2f}")
logger.info("Outstanding taken items:")
for obj_id, data in taken_items.items():
    logger.info(f"{data['person_id']} took {data['class']} at frame {data['frame']} (Box: {data['box']})")
cv2.destroyAllWindows()