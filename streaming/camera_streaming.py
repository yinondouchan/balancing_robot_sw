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
import gi
gi.require_version('Gst', '1.0')

from threading import Condition
from http import server
from gi.repository import Gst, GObject

try:
    resolution = sys.argv[1]
except IndexError:
    resolution = "1280x720"

width, height = resolution.split('x')

try:
    framerate = int(sys.argv[2])
except IndexError:
    framerate = 30

try:
    quality = int(sys.argv[3])
except IndexError:
    quality = 85

aspect_ratio = float(width)/float(height)
window_height = 650
window_width = 16 * window_height / 9

PAGE="""\
<html>
<head>
<title>The all seeing eye of Robot-Aba</title>
</head>
<body>
<center><h1>The all seeing eye of Robot-Aba</h1></center>
<center><video src="stream.mkv" type='video/webm; codecs="theora, vorbis"' width="%s" height="%s"></center>
</body>
</html>
""" % (window_width, window_height)
 
class OpenCVCameraStreaming:
  
    def __init__(self):
        self.condition = Condition()
        self.frame = None

    def video_callback(self, sink):
        with self.condition:             
            # notify new image to all consumers
            sample = sink.emit('pull-sample')
            buf = sample.get_buffer()
            self.frame = buf.extract_dup(0, buf.get_size())
            self.condition.notify_all()

        return Gst.FlowReturn.OK
    
    def run(self):
        Gst.init()
        #gst_str = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=%s, height=%s, format=NV12, framerate=%s/1 ! omxvp8enc bitrate=12000000 ! matroskamux ! appsink emit-signals=true sync=false max-buffers=2 drop=true" % (width, height, framerate)
        gst_str = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=%s, height=%s, format=NV12, framerate=%s/1 ! omxvp8enc bitrate=12000000 ! webmmux ! tcpserversink host=0.0.0.0 port=5000" % (width, height, framerate)
        pipeline = Gst.parse_launch(gst_str)
        pipeline.set_state(Gst.State.PLAYING)
        self.video_sink = pipeline.get_by_name('appsink0')
        self.video_sink.connect('new-sample', self.video_callback)

        print('Camera opened successfully')

            
    
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
        elif self.path == '/stream.mkv':
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
                    self.send_header('Content-Type', 'video/webm')
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
logging.info('Starting server on port %s' % address[1])
server = StreamingServer(address, StreamingHandler)

server.serve_forever()
#finally:
#    output.close()

