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
import argparse
gi.require_version('Gst', '1.0')

from threading import Condition
from http import server
from gi.repository import Gst, GObject
 
class OpenCVCameraStreaming:
  
    def __init__(self, width, height, framerate, quality=85, argus_width=None, argus_height=None):
        self.condition = Condition()
        self.frame = None
        self.width = width
        self.height = height
        self.framerate = framerate
        self.quality = quality
        self.argus_width = argus_width
        self.argus_height = argus_height

        if self.argus_width is None or self.argus_height is None:
            self.argus_width, self.argus_height = self.get_argus_resolution(int(width), int(height), int(framerate))

    def video_callback(self, sink):
        with self.condition:             
            # notify new image to all consumers
            sample = sink.emit('pull-sample')
            buf = sample.get_buffer()
            self.frame = buf.extract_dup(0, buf.get_size())
            self.condition.notify_all()

        return Gst.FlowReturn.OK

    def get_argus_resolution(self, width, height, framerate):
        """
        NVIDIA's argus libraries has some specific resolution and framerate settings.
        Return the setting which brings the best quality given required resolution and framerate.
        """
        if height > 1848:
            return 3280, 2464
        elif height > 720 or framerate <= 28:
            return 3280, 1848
        else:
            # height <= 720 and framerate > 28
            return 1280, 720
        
    def run(self):
        Gst.init()
        #gst_str = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=%s, height=%s, format=NV12, framerate=%s/1 ! omxvp8enc bitrate=12000000 ! matroskamux ! appsink emit-signals=true sync=false max-buffers=2 drop=true" % (width, height, framerate)
        gst_str = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=%s, height=%s, format=NV12, framerate=%s/1 ! nvvidconv ! video/x-raw(memory:NVMM), format=NV12, width=%s, height=%s ! nvjpegenc quality=%s ! appsink emit-signals=true sync=true" % (self.argus_width, self.argus_height, self.framerate, self.width, self.height, self.quality)
        print("Gstreamer string: %s" % gst_str)
        pipeline = Gst.parse_launch(gst_str)
        pipeline.set_state(Gst.State.PLAYING)
        self.video_sink = pipeline.get_by_name('appsink0')
        self.video_sink.connect('new-sample', self.video_callback)

        print('Camera opened successfully')

            
    
    def close(self):
        self._cap.release()

class StreamingHandler(server.BaseHTTPRequestHandler):
    
    _page = None

    @classmethod
    def init_page(cls, width=1280, height=720):
        cls._page="""\
        <html>
        <head>
        <title>The all seeing eye of Robot-Aba</title>
        </head>
        <body>
        <center><h1>The all seeing eye of Robot-Aba</h1></center>
        <center><img src="stream.mjpg" width="%s" height="%s"></center>
        </video>
        </center>
        </body>
        </html>
        """ % (width, height)

    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = self._page.encode('utf-8')
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

args_split = [arg.split('=') for arg in sys.argv[1:]]
args_dict = {arg[0]: arg[1] for arg in args_split}

try:
    width, height = args_dict['res'].split('x')
except KeyError:
    width = 1280
    height = 720

try:
    framerate = args_dict['framerate']
except KeyError:
    framerate = 28

try:
    quality = args_dict['quality']
except KeyError:
    quality = 50

try:
    argus_width, argus_height = args_dict['argus_res'].split('x')
except KeyError:
    argus_width = None
    argus_height = None

try:
    window_width, window_height = args_dict['window_res'].split('x')
except KeyError:
    aspect_ratio = float(width)/float(height)
    window_height=720
    window_width = aspect_ratio * window_height

StreamingHandler.init_page(window_width, window_height)

# streaming output
output = OpenCVCameraStreaming(width, height, framerate, quality=quality, argus_width=argus_width, argus_height=argus_height)
output.run()

#try:
address = ('', 8000)
logging.info('Starting server on port %s' % address[1])
server = StreamingServer(address, StreamingHandler)

server.serve_forever()
#finally:
#    output.close()

