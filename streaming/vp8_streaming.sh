gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! 'video/x-raw(memory:NVMM), width=3280, height=1848, format=NV12, framerate=28/1' ! nvvidconv ! 'video/x-raw(memory:NVMM), format=NV12, width=1920, height=1080' ! omxvp8enc bitrate=8000000 ! webmmux ! tcpserversink host=0.0.0.0 port=5000
