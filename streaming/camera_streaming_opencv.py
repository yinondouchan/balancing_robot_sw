# Web streaming example
# Source code from the official PiCamera package
# http://picamera.readthedocs.io/en/latest/recipes2.html#web-streaming

import io
import logging
import threading
import socketserver
import sys
import cv2
import numpy as np

from threading import Condition
from http import server

try:
    resolution = sys.argv[1]
except IndexError:
    resolution = "320x240"

try:
    framerate = int(sys.argv[2])
except IndexError:
    framerate = 24

PAGE="""\
<html>
<head>
<title>The all seeing eye of Robot-Aba</title>
</head>
<body>
<center><h1>The all seeing eye of Robot-Aba</h1></center>
<center><img src="stream.mjpg" width="800" height="600"></center>
</body>
</html>
"""
    
def gstreamer_pipeline(capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=60, flip_method=0):
        pipeline = ('nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)%s, height=(int)%s, format=(string)NV12, framerate=(fraction)%s/1 ! nvvidconv flip-method=%s ! video/x-raw, width=(int)%s, height=(int)%s, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink'  % (capture_width,capture_height,framerate,flip_method,display_width,display_height))
        return pipeline
 
class OpenCVCameraStreaming(threading.Thread):
  
    def __init__(self):
        self.condition = Condition()
        self.frame = None

        gst_str = "nvarguscamerasrc ! 'video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=60/1' ! nvvidconv flip-method=0 ! 'video/x-raw, width=1280, height=720, format=BGRx' ! videoconvert ! 'video/x-raw, format=BGR' ! appsink"
        gst_str2 = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=60/1 ! nvvidconv flip-method=0 ! video/x-raw, width=1280, height=720, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink emit-signals=true sync=false max-buffers=2 drop=true"
        gst_test = "videotestsrc num-buffers=50 ! appsink emit-signals=true sync=false max-buffers=2 drop=true"

        self._cap = cv2.VideoCapture(gst_str2, cv2.CAP_GSTREAMER)
        if not self._cap.isOpened():
            print('Failed to open camera')
            sys.exit()

        print('Camera opened successfully')
        #self._cap.set(cv2.CAP_PROP_FPS, 60)
        #self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        #self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        threading.Thread.__init__(self)
    
    def run(self):
        logging.info('Starting OpenCV camera streaming thread')
        while True:
            with self.condition:
                # get image
                ret, img = self._cap.read()
                print(ret)
                # encode image to byte array
                self.frame = bytearray(cv2.imencode('.jpeg', img)[1])
                            
                # notify new image to all consumers
                self.condition.notify_all()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    
    def close(self):
        self._cap.release()

class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

# streaming output
output = OpenCVCameraStreaming()
output.run()

#try:
address = ('', 8000)
server = StreamingServer(address, StreamingHandler)

logging.info('Starting server on port %s' % address[1])

server.serve_forever()
#finally:
#    output.close()

