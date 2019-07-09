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
from gstreamer import get_argus_resolution, GStreamerCameraSource, GStreamerConfigs
 
class CameraStreamingSource:
  
    def __init__(self, width, height, framerate, quality=85, argus_width=None, argus_height=None):
        self.condition = Condition()
        self.frame = None
        if argus_width is None or argus_height is None:
            argus_width, argus_height = get_argus_resolution(int(width), int(height), int(framerate))
        self._camera_source = GStreamerCameraSource(GStreamerConfigs.MJpeg(width, height, framerate, argus_width, argus_height))

    def video_callback(self, sink):
        with self.condition:             
            # notify new image to all consumers
            sample = sink.emit('pull-sample')
            buf = sample.get_buffer()
            self.frame = buf.extract_dup(0, buf.get_size())
            self.condition.notify_all()

        return Gst.FlowReturn.OK
        
    def run(self):
        self._camera_source.run_stream(self.video_callback)
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

if __name__ == '__main__':
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
    output = CameraStreamingSource(width, height, framerate, quality=quality, argus_width=argus_width, argus_height=argus_height)
    output.run()

    address = ('', 8000)
    logging.info('Starting server on port %s' % address[1])
    server = StreamingServer(address, StreamingHandler)

    server.serve_forever()

