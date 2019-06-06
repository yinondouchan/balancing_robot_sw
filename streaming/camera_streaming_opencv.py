# Web streaming example
# Source code from the official PiCamera package
# http://picamera.readthedocs.io/en/latest/recipes2.html#web-streaming

import io
import picamera
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

class StreamingOutput(object):
    def __init__(self):
        self.frame = None
        self.buffer = io.BytesIO()
        self.condition = Condition()

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # New frame, copy the existing buffer's content and notify all
            # clients it's available
            self.buffer.truncate()
            with self.condition:
                self.frame = self.buffer.getvalue()
                self.condition.notify_all()
            self.buffer.seek(0)
        return self.buffer.write(buf)
    
    
class OpenCVCameraStreaming(threading.Thread):
    
    def __init__(self):
        self.condition = Condition()
        self.frame = None
        
        self._cap = cv2.VideoCapture(0)
        #self._cap.set(cv2.CAP_PROP_FPS, 24)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        threading.Thread.__init__(self)
    
    def run(self):
        logging.info('Starting OpenCV camera streaming thread')
        while True:
            with self.condition:
                # get image
                ret, img = self._cap.read()
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

"""
with picamera.PiCamera(resolution=resolution, framerate=framerate) as camera:
    output = StreamingOutput()
    #Uncomment the next line to change your Pi's Camera rotation (in degrees)
    #camera.rotation = 90
    camera.start_recording(output, format='mjpeg')
    try:
        address = ('', 8000)
        server = StreamingServer(address, StreamingHandler)
        server.serve_forever()
    finally:
        camera.stop_recording()
"""