'''
Aim: 
	Launch the first cam found, displays the camfeed
	and starts recording.

	Some things are hardcoded: 
 	 codec is DIVX
 	 frame rate is 25
 	 file is written as video.mkv to dir containing this file
 
Command line usage: 
	python recordVideo.py

Notice: 
	No copyrights, warrenty or guarenties are associated with this file.
 
Date: 4Dec2013 

Author: Sameer Khan (samkhan13.wordpress.com)
'''

# Required modules
import cv2
import sys

# Create window for displaying images
cv2.namedWindow("Original")

# Open the video channel 
cap0 = cv2.VideoCapture(-1) # use 1 or 2 or ... for other camera

size = (int(cap0.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
        int(cap0.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
if size == (0,0):
    sys.exit('Camera did not provide frame. Check camera connection')

# Settings for recording the video
codec = cv2.cv.CV_FOURCC('D','I','V','X')
fps = 25
    
# Start writing to a file on disk
videoFile = cv2.VideoWriter()
videoFile.open('video.mkv', codec, fps, size, 1)
print 'Created video file, started recording...'
print 'Press any key to finish recording'

key = -1
while(key < 0):
    success0, img0 = cap0.read()
    cv2.imshow("Original", img0)
    videoFile.write(img0)
    key = cv2.waitKey(1)
    
# Close video channel
cap0.release()

cv2.destroyAllWindows()
#cv.DestroyWindow('Original')

print 'Finished recording'