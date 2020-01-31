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

class FrameSource:

    DIM = (1280, 720)
    K = np.array(
        [[603.2367723329799, 0.0, 631.040490386724], [0.0, 606.6503954431822, 394.50828189919275], [0.0, 0.0, 1.0]])
    D = np.array([[-0.06211310762499229], [0.11678409244618092], [-0.20084647516958823], [0.10142080878217873]])

    def undistort_balanced(self, img, balance=0.0, dim2=None, dim3=None):
        dim1 = img.shape[:2][::-1]  # dim1 is the dimension of input image to un-distort
        assert dim1[0] / dim1[1] == DIM[0] / DIM[
            1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
        if not dim2:
            dim2 = dim1
        if not dim3:
            dim3 = dim1
        scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
        scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
        # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return undistorted_img

    """ Feed this class with frames from a specific source """
    def __init__(self):
        self.frame = None
        self.frame_mutex = Condition()

    def on_frame(self, frame):
        with self.frame_mutex:
            self.frame = frame
            self.frame_mutex.notify_all()
    

class StreamingHandler(server.BaseHTTPRequestHandler):
    """ handles http requests for different endpoints """
    
    _page = None
    _frame_source = None

    @classmethod
    def init_page(cls, width=1280, height=720):
        cls._page="""\
        <html>
        <head>
        <title>Yep, it can see</title>
        </head>
        <body>
        <center><h1>The all seeing eye of my fabulous robot</h1></center>
        <center><img src="stream.mjpg" width="%s" height="%s"></center>
        </video>
        </center>
        </body>
        </html>
        """ % (width, height)
        
    @classmethod
    def set_frame_source(cls, frame_source):
        cls._frame_source = frame_source

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
                    frame_source = self._frame_source
                    with frame_source.frame_mutex:
                        frame_source.frame_mutex.wait()
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'video/webm')
                    self.send_header('Content-Length', len(frame_source.frame))
                    self.end_headers()
                    self.wfile.write(frame_source.frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    """ the streaming server """
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
        
    try:
        output_src = args_dict['src']
    except KeyError:
        output_src = 'camera'
        
    
    if argus_width is None or argus_height is None:
        argus_width, argus_height = get_argus_resolution(int(width), int(height), int(framerate))

    StreamingHandler.init_page(window_width, window_height)

    # streaming output
    frame_source = FrameSource()
    StreamingHandler.set_frame_source(frame_source)
    
    if output_src == 'camera':
        output = GStreamerCameraSource(GStreamerConfigs.MJpeg(width, height, framerate, quality=quality, argus_width=argus_width, argus_height=argus_height),
                                        frame_source.on_frame)
    elif output_src == 'shared_memory':
        try:
            shared_memory_path = args_dict['src_path']
        except KeyError:
            raise Exception('src_path must be determined if using shared memory')
            
        output = GStreamerCameraSource(GStreamerConfigs.SharedMemoryToMJPeg(src_path=shared_memory_path, quality=quality),
                                        frame_source.on_frame)
    else:
        raise Exception('Unknown source: %s' % output_src)
        

    address = ('', 8000)
    logging.info('Starting server on port %s' % address[1])
    server = StreamingServer(address, StreamingHandler)

    server.serve_forever()

