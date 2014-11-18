# Required modules
import cv2
import imgUtils
from numpy import copy
from pylab import arange, figure, plot, show, sort

# Load image in grayscale
firstImg = cv2.imread('./AF23NES.jpg', 0)
secondImg = copy(firstImg)

# Find face in image
haarFaceCascade = imgUtils.chooseCascade() # choseCascade without choice gives frontalFace

firstFaceRegion, firstImg = imgUtils.cropToObj(haarFaceCascade, firstImg)
secondFaceRegion, secondImg = imgUtils.cropToObj(haarFaceCascade, secondImg)

# cropToObj returns a list of objects
firstFaceRegion = firstFaceRegion[0]
secondFaceRegion = secondFaceRegion[0]
firstImg = firstImg[0]
secondImg = secondImg[0]

# Resize to a suitable shape
imgSize = (200,200)

firstImg = cv2.resize(firstImg,imgSize,cv2.cv.CV_INTER_AREA)
# secondImg = cv2.resize(secondImg,imgSize,cv2.cv.CV_INTER_AREA)

print 'firstImg size = {0}'.format(firstImg.shape)
print 'secondImg size = {0}'.format(secondImg.shape)
print '\n'

# Remove forehead and ears
firstImg = imgUtils.cropByPercent(firstImg,0.2,0.2)
secondImg = imgUtils.cropByPercent(secondImg,0.2,0.2)

# Choose an algorithm for extracting keypoints
#detectAlgo = 'SURF'
#featureExtractor = cv2.FeatureDetector_create(detectAlgo)

featureExtractor = cv2.SURF(hessianThreshold = 40, nOctaves = 3, nOctaveLayers = 3, extended = True, upright = True)

firstKP = featureExtractor.detect(firstImg, mask = None)
secondKP = featureExtractor.detect(secondImg, mask = None)

print 'firstKP = {0}'.format(len(firstKP))
print 'secondKP = {0}'.format(len(secondKP))
print '\n'

# Choose an algorithm to describe the keypoints
descriptor = cv2.DescriptorExtractor_create('SURF')

firstKP, firstDesc = descriptor.compute(firstImg,firstKP)
secondKP, secondDesc = descriptor.compute(secondImg,secondKP)


# Chose an algorithm to match descroptors
# Options: BruteForce, BruteForce-L1, BruteForce-Hamming, BruteForce-Hamming(2), FlannBased        
matcher = cv2.DescriptorMatcher_create('FlannBased')

matches = matcher.match(firstDesc, secondDesc)
print '>> matches:', len(matches)

# Choose a threshold for selecting "good" matches
dist = [m.distance for m in matches]
#thresh_dist = (sum(dist) / len(dist)) # mean distance as threshold
thresh_dist = max(dist)*0.6

print 'distance: min: %.3f' % min(dist)
print 'distance: mean: %.3f' % (sum(dist) / len(dist))
print 'distance: max: %.3f' % max(dist)

good_matches = [m for m in matches if m.distance <= thresh_dist]
poor_matches = [m for m in matches if m.distance > thresh_dist]

print '>> selected matches:', len(good_matches)

# Visualization
goodMatchImage = imgUtils.drawMatches(firstImg,firstKP,secondImg,secondKP,good_matches)
poorMatchImage = imgUtils.drawMatches(firstImg,firstKP,secondImg,secondKP,poor_matches)

cv2.imshow("goodMatchImage", goodMatchImage)
cv2.imshow("poorMatchImage", poorMatchImage)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imwrite('./SURF_goodMatches1.jpg',goodMatchImage)
cv2.imwrite('./SURF_poorMatches1.jpg',poorMatchImage)

figure()
plot(arange(len(dist)),sort(dist)) # plot(xData,yData)
show()