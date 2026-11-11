import os,sys
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

# Avoid OpenCV worker oversubscription on constrained free servers / CI.
try:
    import cv2
    cv2.setNumThreads(1)
except ImportError:
    pass
