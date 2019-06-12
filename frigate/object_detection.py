import datetime
import time
import cv2
import threading
import numpy as np
#from edgetpu.detection.engine import DetectionEngine
from openvino.inference_engine import IENetwork, IEPlugin
from . util import tonumpyarray

PATH_TO_MODEL_XML = './model/model.xml'
PATH_TO_MODEL_BIN = './model/model.bin'
PATH_TO_MODEL_MAPPING = './model/model.mapping'
PATH_TO_MODEL_LABELS = './model/model.labels'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
#PATH_TO_CKPT = '/frozen_inference_graph.pb' 
# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = '/label_map.pbtext'

# Function to read labels from text files.
def ReadLabelFile(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    ret = {}
    for line in lines:
        pair = line.strip().split(maxsplit=1)
        ret[int(pair[0])] = pair[1].strip()
    return ret

class PreppedQueueProcessor(threading.Thread):
    def __init__(self, cameras, prepped_frame_queue):

        threading.Thread.__init__(self)
        self.cameras = cameras
        self.prepped_frame_queue = prepped_frame_queue
        
        # Load the edgetpu engine and labels
        #self.engine = DetectionEngine(PATH_TO_CKPT)
        #self.labels = ReadLabelFile(PATH_TO_LABELS)

        model_xml = PATH_TO_MODEL_XML
        model_bin = PATH_TO_MODEL_BIN
        labels = PATH_TO_MODEL_LABELS
        # Plugin initialization for specified device and load extensions library if specified

        self.plugin = IEPlugin(device='MYRIAD', plugin_dirs=None)
        self.net = IENetwork(model=model_xml, weights=model_bin)
        assert len(net.inputs.keys()) == 1, "Demo supports only single input topologies"
        assert len(net.outputs) == 1, "Demo supports only single output topologies"
        self.input_blob = next(iter(net.inputs))
        self.out_blob = next(iter(net.outputs))
        self.exec_net = plugin.load(network=self.net, num_requests=2)
        #n, c, h, w = net.inputs[self.input_blob].shape
        del net
        with open(labels, 'r') as f:
            self.labels_map = [x.strip() for x in f]
        self.next_request_id = 1
        self.cur_request_id = 0

    def run(self):
        # process queue...
        while True:
            frame = self.prepped_frame_queue.get()

            # Actual detection.
            #objects = self.engine.DetectWithInputTensor(frame['frame'], threshold=frame['region_threshold'], top_k=3)
            inf_start = time.time()
            self.exec_net.start_async(request_id=next_request_id, inputs={input_blob: frame})
            if self.exec_net.requests[cur_request_id].wait(-1) == 0 :
                inf_end = time.time()
                det_time = inf_end - inf_start

            # Parse detection results of the current request
                res = self.exec_net.requests[cur_request_id].outputs[self.out_blob]
            # parse and pass detected objects back to the camera
            parsed_objects = []

            for obj in res[0][0]:
                # Add objects when probability more than specified threshold
                if obj[2] > frame['region_threshold'] :
                    class_id = int(obj[1])
                    det_label = labels_map[class_id] if labels_map else str(class_id)
                    parsed_objects.append({
                        'frame_time': frame['frame_time'],
                        'name': str(det_label),
                        'score': float(obj[2]),
                        'xmin' : int((obj[3] * frame['region_size']) + frame['region_x_offset']),
                        'ymin' : int((obj[4] * frame['region_size']) + frame['region_x_offset']),
                        'xmax' : int((obj[5] * frame['region_size']) + frame['region_x_offset']),
                        'ymax' : int((obj[6] * frame['region_size']) + frame['region_x_offset'])
                    })
                
#            for obj in objects:
#                box = obj.bounding_box.flatten().tolist()
#                parsed_objects.append({
#                            'frame_time': frame['frame_time'],
#                            'name': str(self.labels[obj.label_id]),
#                            'score': float(obj.score),
#                            'xmin': int((box[0] * frame['region_size']) + frame['region_x_offset']),
#                            'ymin': int((box[1] * frame['region_size']) + frame['region_y_offset']),
#                            'xmax': int((box[2] * frame['region_size']) + frame['region_x_offset']),
#                            'ymax': int((box[3] * frame['region_size']) + frame['region_y_offset'])
#                        })

            self.cameras[frame['camera_name']].add_objects(parsed_objects)


# should this be a region class?
class FramePrepper(threading.Thread):
    def __init__(self, camera_name, shared_frame, frame_time, frame_ready, 
        frame_lock,
        region_size, region_x_offset, region_y_offset, region_threshold,
        prepped_frame_queue):

        threading.Thread.__init__(self)
        self.camera_name = camera_name
        self.shared_frame = shared_frame
        self.frame_time = frame_time
        self.frame_ready = frame_ready
        self.frame_lock = frame_lock
        self.region_size = region_size
        self.region_x_offset = region_x_offset
        self.region_y_offset = region_y_offset
        self.region_threshold = region_threshold
        self.prepped_frame_queue = prepped_frame_queue

    def run(self):
        frame_time = 0.0
        while True:
            now = datetime.datetime.now().timestamp()

            with self.frame_ready:
                # if there isnt a frame ready for processing or it is old, wait for a new frame
                if self.frame_time.value == frame_time or (now - self.frame_time.value) > 0.5:
                    self.frame_ready.wait()
            
            # make a copy of the cropped frame
            with self.frame_lock:
                cropped_frame = self.shared_frame[self.region_y_offset:self.region_y_offset+self.region_size, self.region_x_offset:self.region_x_offset+self.region_size].copy()
                frame_time = self.frame_time.value
            
            # convert to RGB
            cropped_frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            cropped_frame_rgb= cropped_frame_rgb.transpose((2, 0, 1))
            # Resize to 300x300 if needed
            if cropped_frame_rgb.shape != (300, 300, 3):
                cropped_frame_rgb = cv2.resize(cropped_frame_rgb, dsize=(300, 300), interpolation=cv2.INTER_LINEAR)
            # Expand dimensions since the model expects images to have shape: [1, 300, 300, 3]
            frame_expanded = np.expand_dims(cropped_frame_rgb, axis=0)

            # add the frame to the queue
            if not self.prepped_frame_queue.full():
                self.prepped_frame_queue.put({
                    'camera_name': self.camera_name,
                    'frame_time': frame_time,
                    'frame': frame_expanded.flatten().copy(),
                    'region_size': self.region_size,
                    'region_threshold': self.region_threshold,
                    'region_x_offset': self.region_x_offset,
                    'region_y_offset': self.region_y_offset
                })
            else:
                print("queue full. moving on")
