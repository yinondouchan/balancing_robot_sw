import gi
cv2 = None

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject

def get_argus_resolution(width, height, framerate):
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

class GStreamerConfigs:
    
    class SharedMemoryToMJPeg:
        def __init__(self, src_path=None, quality=50):
            self.gstreamer_str = """ shmsrc socket-path=%s ! video/x-raw,format=I420,width=800,height=450,framerate=60/1 ! nvjpegenc ! appsink emit-signals=true sync=true """ % src_path
    
    class RawAppsinkToMJPeg:
        def __init__(self, quality=50):
            self.quality = quality
            
            self.gstreamer_str = """ appsrc ! nvvidconv ! nvjpegenc quality=%s ! appsink emit-signals=true sync=true """ % self.quality
	
    class MJpeg:
        """ mjpeg directed to appsink """
        def __init__(self, width=1280, height=720, framerate=30, argus_width=3280, argus_height=1848, quality=50, flip_method=0, format_='NV12'):
            self.width = width
            self.height = height
            self.framerate = framerate
            self.argus_width = argus_width
            self.argus_height = argus_height
            self.quality = quality
            self.flip_method = flip_method
            self.format_ = format_
			
            self.gstreamer_str = """nvarguscamerasrc ! video/x-raw(memory:NVMM), width=%s, height=%s, format=NV12,
			 						framerate=%s/1 ! nvvidconv flip_method=%s ! video/x-raw(memory:NVMM), format=%s,
			 						width=%s, height=%s ! nvjpegenc quality=%s ! appsink emit-signals=true sync=true"""	% (self.argus_width, self.argus_height, self.framerate, self.flip_method, self.format_, self.width, self.height, self.quality)
                                    
    class MJpegTCP:
        """ mjpeg directed to tcpserversink """
        def __init__(self, width=1280, height=720, framerate=30, argus_width=3280, argus_height=1848, quality=50, flip_method=0):
            self.width = width
            self.height = height
            self.framerate = framerate
            self.argus_width = argus_width
            self.argus_height = argus_height
            self.quality = quality
            self.flip_method = flip_method
			
            self.gstreamer_str = """nvarguscamerasrc ! video/x-raw(memory:NVMM), width=%s, height=%s, format=NV12,
			 						framerate=%s/1 ! nvvidconv flip_method=%s ! video/x-raw(memory:NVMM), format=NV12,
			 						width=%s, height=%s ! nvjpegenc quality=%s ! tcpserversink host=0.0.0.0 port=5000""" % (self.argus_width, self.argus_height, self.framerate, self.flip_method, self.width, self.height, self.quality)
				
    class Direct:
        """ no encoding, directly to appsink """
        def __init__(self, width=1280, height=720, framerate=30, argus_width=3280, argus_height=1848, flip_method=0):
            self.width = width
            self.height = height
            self.framerate = framerate
            self.argus_width = argus_width
            self.argus_height = argus_height
            self.flip_method = flip_method
			
            self.gstreamer_str = """nvarguscamerasrc ! video/x-raw(memory:NVMM), width=%s, height=%s, format=NV12,
			 						framerate=%s/1 ! nvvidconv flip_method=%s ! video/x-raw, format=BGRx,
			 						width=%s, height=%s ! videoconvert ! appsink emit-signals=true sync=true""" 	% (self.argus_width, self.argus_height, self.framerate, self.flip_method, self.width, self.height)
                                    
    vp8_enc_gst_str = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=3280, height=1848, format=NV12, framerate=28/1 ! omxvp8enc bitrate=12000000 ! matroskamux ! appsink emit-signals=true sync=false max-buffers=2 drop=true"


class GStreamerCameraSource:
	
    def __init__(self, config=None, frame_cb=None):
        self._config = config
    
        if self._config is None:
            self._config = GStreamerConfigs.MJpeg()
            
        self._frame_cb = frame_cb
            
        self.init_stream()
				
    def init_stream(self):
        Gst.init()
        pipeline = Gst.parse_launch(self._config.gstreamer_str)
        pipeline.set_state(Gst.State.PLAYING)
        self.video_sink = pipeline.get_by_name('appsink0')
        self.video_sink.connect('new-sample', self.video_callback)
        
    def video_callback(self, sink):
        """ called when a new frame arrives """ 
        # extract frame buffer from gstreamer          
        sample = sink.emit('pull-sample')
        buf = sample.get_buffer()
        frame = buf.extract_dup(0, buf.get_size())
        if self._frame_cb:
            self._frame_cb(frame)

        return Gst.FlowReturn.OK
 
    def close(self):
        self._cap.release()
        
    #def to_opencv_videocapture(self):
    #    if cv2 is None:
    #        import cv2
    #        
    #    return cv2.VideoCapture(self._config.gstreamer_str, cv2.CAP_GSTREAMER)
        
    
class GStreamerCameraSink:
    def __init__(self):
        pass
        
    def send_frame(self, frame):
        """ send a frame to the appsrc """
