'''
These functions are most often utilized in a signal processing pipline
and are organized within this module to reprsent the form:

signal --> acquire as data --> processing --> output as data
'''

# TODO: Import only specific things and not the entire namespace
# TODO: What if the namespace is already imported? 

import os
import cv2
import numpy as np
import cPickle

#===============================================================================
# Acquire data
#===============================================================================

# facerec predictor loaded from pickled file   
def loadPredictor(fileName):
    if not fileName:
        print 'Failed to load: given file name was None type'
    else:
        pklFile = open(fileName, 'rb')
        predictor = cPickle.load(pklFile)
        pklFile.close()
        return predictor

# Files from a folder with subfolders
def fetchFiles(pathToFolder, flag, keyWord):
    # TODO: Display directory structure using dirPath, dirNames
    '''fetchFiles() requires three arguments:
     
    1. pathToFolder is the location of files which may contain subfolders
    2. flag must be 'STARTS_WITH' or 'ENDS_WITH'
    3. keyWord is a string to search the file's name
    
    Be careful, the keyWord is case sensitive and must be exact
    
    Example: fetchFiles('/Documents/Photos/','ENDS_WITH','.jpg')
        
    returns: _pathToFiles, _fileNames '''
    
    _pathToFiles = []
    _fileNames = []

    for dirPath, dirNames, fileNames in os.walk(pathToFolder):
        if flag == 'ENDS_WITH':
            selectedPath = [os.path.join(dirPath,item) for item in fileNames 
                            if item.endswith(keyWord)]     
            _pathToFiles.extend(selectedPath)
            
            selectedFile = [item for item in fileNames if 
                            item.endswith(keyWord)]
            _fileNames.extend(selectedFile)
            
        elif flag == 'STARTS_WITH':
            selectedPath = [os.path.join(dirPath,item) for item in fileNames 
                            if item.startswith(keyWord)]
            _pathToFiles.extend(selectedPath)
            
            selectedFile = [item for item in fileNames 
                            if item.startswith(keyWord)]
            _fileNames.extend(selectedFile) 
                
        else:
            print fetchFiles.__doc__
            break
                        
        # Try to remove empty entries if no required files are in directory
        try:
            _pathToFiles.remove('')
            _fileNames.remove('')
        except ValueError:
            pass
            
        # Warn if nothing was found amongst the given paths
        if selectedFile == []: 
            print 'No files with given parameters were found in:\n', dirPath, '\n'
        
    print len(_fileNames), 'files were found in searched folder(s)'
                
    return _pathToFiles, _fileNames
    
#===============================================================================
# Process the acquired data
#===============================================================================

def isEmpty(anyStructure):
    if anyStructure:
        return False
    else:
        return True
    
def chooseCascade(choice='frontalFace'):
    # TODO: write the doc for this function
    
    # 'mouth' cascade doesn't work properly
    opts = {'frontalFace':'./cascadeFiles/haarcascade_frontalface_alt2.xml',
    			'profileFace':'./cascadeFiles/haarcascade_profileface.xml',
    			'leftEye':'./cascadeFiles/haarcascade_leftEye.xml',
    			'rightEye':'./cascadeFiles/haarcascade_rightEye.xml',
    			'rightEye2':'./cascadeFiles/haarcascade_righteye_2splits.xml',
    			'frontalEyes':'./cascadeFiles/haarcascade_frontalEyes.xml',
    			'eyeGlasses':'./cascadeFiles/haarcascade_eye_tree_eyeglasses.xml',
    			'fullBody':'./cascadeFiles/haarcascade_eye_tree_eyeglasses.xml',
    			'upperBody':'./cascadeFiles/haarcascade_upperbody.xml',
    			'lowerBody':'./cascadeFiles/haarcascade_upperbody.xml',
    			'lowerBody':'./cascadeFiles/haarcascade_lowerbody.xml',
    			'head':'./cascadeFiles/haarcascade_head_16x16.xml',
    			'nose':'./cascadeFiles/haarcascade_nose_25x15.xml',
    			'mouth':'./cascadeFiles/haarcascade_mcs_mouth.xml',   			
                'frontalFace_LBP':'./cascadeFiles/lbpcascade_frontalface.xml',
    			'mascfrontalFace_LBP': './cascadeFiles/lbpcascade_mascFrontalFace_20x20_Nov-22-2013.xml'}
    
    _cascade = cv2.CascadeClassifier(opts[choice])   
    return _cascade

def cropToObj(cascade,image):
    '''
    Crop to the object of interest in the image
    
    Since cascade.detectMultiScale() can return a list
    the output of this function is also a list
    even for the case when a single image is to be returned
    '''
    
    # TODO: What if croping shouldn't produce rectangular region?
    # TODO: What if objImages keeps growing and consumes all available memory?

    _objRegions = cascade.detectMultiScale(image,
                                           scaleFactor=1.2, 
                                           minNeighbors=3, 
                                           minSize=(20, 20)) 
    
#     _objRegions = cascade.detectMultiScale(image)
    
    _objImages = []
    for rowNum in range(len(_objRegions)):
        x1 = _objRegions[rowNum,0]
        y1 = _objRegions[rowNum,1]
        x1PlusWidth = _objRegions[rowNum,0] + _objRegions[rowNum,2]
        y1PlusHeight = _objRegions[rowNum,1] + _objRegions[rowNum,3]
    
        _objImage = image[y1:y1PlusHeight,x1:x1PlusWidth]
        _objImages.append(_objImage)
    
    return _objRegions, _objImages

def cropByPercent(_image,hAxisPct,vAxisPct):
    ''' 
    use only for face images returned by cropToObj!
    
    crop off the forehead by hAxisPct
    crop off the ears by vAxisPct
    
    hAxisPct and vAxisPct should be a value between 0 and 1
    
    example: 
    image2 = cropByPercent(image1,0.3,0.2)
    '''
    hAxis = int(hAxisPct*_image.shape[0])
    vAxis = int(vAxisPct*_image.shape[1])
            
    croppedImage = _image[hAxis:, vAxis:-vAxis]
            
    return croppedImage


#===============================================================================
# Output processed data
#===============================================================================

# Open the main camera and save an image
def saveCamImage(path):
    '''
    Give destination path as string
    Open primary camera
    Display the image in a window
    Press s to save the image as 'camImage.jpg'
    Press esc to quit
    '''
    capture = cv2.VideoCapture(0)
    
    saveCount = 0
    keyPressed = -1
    while(keyPressed != 27):
        ret, camImage = capture.read()
        cv2.imshow("camImage", camImage)
        
        keyPressed = cv2.waitKey(1)
        
        if keyPressed == ord('s'):
            cv2.imwrite(path + 'camImage' + str(saveCount) + '.jpg', camImage)
            saveCount += 1
            print 'image saved to workspace'
    cv2.destroyAllWindows()

# Save facerec predictor to pickle file
def savePredictor(fileName, predictor):
    if not predictor:
        print 'Failed to save: The given predictor was None type.'
    else:
        pklFile = open(fileName, 'wb')
        cPickle.dump(predictor, pklFile)
        pklFile.close()

# A text file with paths to desired data    
def writePathsToFile(outputFileName,fileNames,prefix='',suffix=''):
    '''
    All arguments must be strings
    
    get pathToFiles and fileNames via fetchFiles()
    or supply them via other suitable means
    
    prefix and suffix are optional
    
    Example: 
    writePathsToFiles('output.txt', 
                      'abc123.jpg', 
                      '/traindata ', 
                      ' 1 0 0 240 240') 
    
    The line written in output.txt will be:
    /traindata abc123.jpg 1 0 0 240 240
    '''
    
    textFile = open(outputFileName,'w')

    for _name in fileNames:
        textFile.write(prefix + _name + suffix +'\n')
        
    textFile.close()
 
# for writing color image to grayscale   
def toGrayScale(pathToFiles,dstPath):
    '''
    get pathToFiles and fileNames via fetchFiles()
    or supply them via other suitable means
    
    saves a 1-channel 8bit grayscale image at dstPath.
    '''
    
    for path in pathToFiles:
        grayImage = cv2.imread(path,0)
    
        cv2.imwrite(dstPath,grayImage)
        
def drawMatches(img1,kp1,img2,kp2,matches):
    '''
    compare img1 and img2 for matching keypoints
    get matches from KNN or Brute Force Matcher or FLANN
    '''
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    view = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    view[:h1, :w1, 0] = img1
    view[:h2, w1:, 0] = img2
    view[:, :, 1] = view[:, :, 0]
    view[:, :, 2] = view[:, :, 0]
    
    for m in matches:
        # draw the keypoints
        color = tuple([np.random.randint(0, 255) for item in xrange(3)])
        
        pt1 = (int(kp1[m.queryIdx].pt[0]),int(kp1[m.queryIdx].pt[1]))
        pt2 = (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1]))
        
        cv2.line(view, pt1, pt2, color)
    
    return view