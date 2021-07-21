# This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

# Usage example:  python3 object_detection_yolo.py --video=run.mp4
#                 python3 object_detection_yolo.py --image=bird.jpg

import cv2 as cv
import numpy as np
from .predict import predict
from .color import color_predict
import time
# Initialize the parameters
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold

inpWidth = 416  # 608     #Width of network's input image
inpHeight = 416  # 608     #Height of network's input image

# Load names of classes
classesFile = "model/custom/classes.names"

classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.

modelConfiguration = "model/custom/yolov3.cfg"
modelWeights = "model/custom/weights/yolov3_final.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Get the names of the output layers


def predictPlate(image, video, model):
    start_time = time.time()
    plate = []
    image = 'static/images/'+image
    cv.imwrite('static/images/vehicle.jpg', cv.imread(image))
    image = 'static/images/vehicle.jpg'

    def getOutputsNames(net):
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Draw the predicted bounding box

    def drawPred(classId, conf, left, top, right, bottom):
        plate = frame[top:bottom, left:right]
        stime = time.time()
        prediction = predict(plate, model) + color_predict(
            cv.cvtColor(plate, cv.COLOR_BGR2RGB))
        print(prediction)
        # cv.imwrite('static/images/plate.jpg', plate.astype(np.uint8))
        print('After prediciton process:', time.time()-start_time)
        cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        label = '%.2f' % conf

        # Get the label for the class name and its confidence
        # if classes:
        #     assert(classId < len(classes))
        #     label = '%s:%s' % (classes[classId], label)

        # Display the label at the top of the bounding box
        # labelSize, baseLine = cv.getTextSize(
        #     label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(
        #     1.5*labelSize[0]), top + baseLine), (0, 0, 255), cv.FILLED)
        # cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine),    (255, 255, 255), cv.FILLED)
        # cv.putText(frame, label, (left, top),
        #            cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        prediction = prediction + (label,)

        return prediction

    # Remove the bounding boxes with low confidence using non-maxima suppression

    def postprocess(frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            # print("out.shape : ", out.shape)
            for detection in out:
                # if detection[4]>0.001:
                scores = detection[5:]
                classId = np.argmax(scores)
                # if scores[classId]>confThreshold:
                confidence = scores[classId]
                # if detection[4] > confThreshold:
                #     print(detection[4], " - ", scores[classId],
                #           " - th : ", confThreshold)
                #     print(detection)
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv.dnn.NMSBoxes(
            boxes, confidences, confThreshold, nmsThreshold)
        output = [()]

        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            if width and width * height > box[2] * box[3]:
                break
            width = box[2]
            height = box[3]

            output[0] = (drawPred(classIds[i], confidences[i], left,
                                   top, left + width, top + height))
        return output

    if (image):
        # print(len(image))
        cap = cv.VideoCapture(image)
        print('boke h222eare')
        outputFile = image[:-4]+'_yolo_out_py.jpg'
    elif (video):
        cap = cv.VideoCapture(video)
        outputFile = video[:-4]+'_yolo_out_py.avi'

    # Get the video writer initialized to save the output video
    if (not image):
        vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (round(
            cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

    while True:
        # get frame from the video
        hasFrame, frame = cap.read()

        # Stop the program if reached end of video
        if not hasFrame:
            # print("Done processing !!!")
            break

        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(
            frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))
        # Remove the bounding boxes with low confidence
        print('Before prediciton process:', time.time()-start_time)
        output = postprocess(frame, outs)

        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (
            t * 1000.0 / cv.getTickFrequency())
        # cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        # Write the frame with the detection boxes
        if (image):
            cv.imwrite('static/images/temp.jpg', frame.astype(np.uint8))
            if(len(output)):
                print(output)
                return output[0]
            else:
                raise Exception('No plate found')
        else:
            vid_writer.write(frame.astype(np.uint8))
