"""
Copyright 2017-2018 Fizyr (https://fizyr.com)
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
#Editted by Mohammad Rahimzadeh (mr7495@yahoo.com)
from .anchors import compute_overlap
from .visualization import draw_detections, draw_annotations,draw_box,draw_caption
from keras import backend
import keras
import numpy as np
import os

import cv2
import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(generator, model, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the model using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Running network: '):
        raw_image    = generator.load_image(i)
        image        = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        # run network
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]

        # correct boxes for image scale
        boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes      = boxes[0, indices[scores_sort], :]
        image_scores     = scores[scores_sort]
        image_labels     = labels[0, indices[scores_sort]]
        image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        if save_path is not None:
            draw_annotations(raw_image, generator.load_annotations(i), label_to_name=generator.label_to_name)
            draw_detections(raw_image, image_boxes, image_scores, image_labels, label_to_name=generator.label_to_name)

            cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]

    return all_detections


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]
    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Parsing annotations: '):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_annotations[i][label] = annotations['bboxes'][annotations['labels'] == label, :].copy()

    return all_annotations


def evaluate(
    generator,
    model,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    save_path=None,
    save_log=False
):
    """ Evaluate a given dataset using a given model.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    output={}
    all_detections     = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections)
    all_annotations    = _get_annotations(generator)
    all_detections1= _get_detections(generator, model, score_threshold=0.5, max_detections=max_detections)
    average_precisions = {}
    if save_path is None:
        try:
            os.mkdir('Validation_Results/')
        except:
            pass
        finally:
            save_path = 'Validation_Results/'
        
    # all_detections = pickle.load(open('all_detections.pkl', 'rb'))
    # all_annotations = pickle.load(open('all_annotations.pkl', 'rb'))
    # pickle.dump(all_detections, open('all_detections.pkl', 'wb'))
    # pickle.dump(all_annotations, open('all_annotations.pkl', 'wb'))

    # process detections and annotations
    for label in range(generator.num_classes()):
        if not generator.has_label(label):
            continue

        false_positives = np.zeros((0,))
        false_positives_i = np.zeros((0,))
        true_positives  = np.zeros((0,))
        true_positives_i  = np.zeros((0,))
        false_negatives = np.zeros((0,))
        false_negatives_i = np.zeros((0,))
        false_positivesx = np.zeros((0,))
        false_positives_ix = np.zeros((0,))
        true_positivesx  = np.zeros((0,))
        true_positives_ix  = np.zeros((0,))
        false_negativesx = np.zeros((0,))
        false_negatives_ix = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            output[generator.image_names[i]]={'annotations':[],'TP':[],'FP':[],'FN':[]}
            detections           = all_detections[i][label]
            detections1          = all_detections1[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []
            detected_annotations1 = []
            image1 = generator.load_image(i)[:,:,1].astype(np.uint8)
            image1 = cv2.cvtColor(image1,cv2.COLOR_GRAY2BGR)
            
            draw_annotations(image1, generator.load_annotations(i))
            output[generator.image_names[i]]['annotations'].append(generator.load_annotations(i))


            for d in detections:
                scores = np.append(scores, d[4])
                image_boxes1 = d
                image_scores1 = d[4]

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    false_positives_i = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    true_positives_i  = np.append(true_positives, 0)
                    continue
 

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]
            
                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    false_positives_i = np.append(false_positives_i, 0)
                    true_positives  = np.append(true_positives, 1)
                    true_positives_i  = np.append(true_positives_i, 1)
                    detected_annotations.append(assigned_annotation)
                    if image_scores1 >= iou_threshold :
                        false_positivesx = np.append(false_positivesx, 0)
                        false_positives_ix = np.append(false_positives_ix, 0)
                        true_positivesx  = np.append(true_positivesx, 1)
                        true_positives_ix  = np.append(true_positives_ix, 1)
                        draw_box(image1,image_boxes1,color=(0,255,0))
                        draw_caption(image1,image_boxes1,'TP:'+str(round(image_scores1,2)))
                        output[generator.image_names[i]]['TP'].append([image_boxes1,'TP:'+str(round(image_scores1,2))])
                        detected_annotations1.append(assigned_annotation)

                else:
                    false_positives = np.append(false_positives, 1)
                    false_positives_i = np.append(false_positives_i, 1)
                    true_positives  = np.append(true_positives, 0)
                    true_positives_i  = np.append(true_positives_i, 0)
                    if image_scores1 >= iou_threshold :
                        false_positivesx = np.append(false_positivesx, 1)
                        false_positives_ix = np.append(false_positives_ix, 1)
                        true_positivesx  = np.append(true_positivesx, 0)
                        true_positives_ix  = np.append(true_positives_ix, 0)
                        draw_box(image1,image_boxes1,color=(0,0,255))
                        draw_caption(image1,image_boxes1,'FP:'+str(round(image_scores1,2)))
                        output[generator.image_names[i]]['FP'].append([image_boxes1,'FP:'+str(round(image_scores1,2))])
            anno = annotations.tolist()   
            for a in anno:
                if anno.index(a) not in detected_annotations:
                    false_negatives = np.append(false_negatives, 1)
                    false_negatives_i = np.append(false_negatives_i, 1)
                    
            for a in anno:
                if anno.index(a) not in detected_annotations1:
                    image_boxes3=a
                    draw_box(image1,image_boxes3,color=(120,0,120))
                    draw_caption(image1,image_boxes3,'FN')
                    output[generator.image_names[i]]['FN'].append([image_boxes3,'FN'])
                    false_negativesx = np.append(false_negativesx, 1)
                    false_negatives_ix = np.append(false_negatives_ix, 1)
                    
            if len(false_negatives_i) == 0:
                false_negatives1_i = 0
            else:
                false_negatives1_i = int(max(np.cumsum(false_negatives_i)))
                
            if len(false_negatives_ix) == 0:
                false_negatives1_ix = 0
            else:
                false_negatives1_ix = int(max(np.cumsum(false_negatives_ix)))
                
            false_positives_i = np.cumsum(false_positives_i)
            true_positives_i  = np.cumsum(true_positives_i)
            try:
                false_positives1_i = int(max(false_positives_i))
            except:
                false_positives1_i=0
            try:
                 true_positives1_i = int(max(true_positives_i))
            except:
                pass
            false_positives_ix = np.cumsum(false_positives_ix)
            true_positives_ix  = np.cumsum(true_positives_ix)
            try:
                false_positives1_ix = int(max(false_positives_ix))
            except:
                false_positives1_ix=0
            try:
                true_positives1_ix = int(max(true_positives_ix))
            except:
                true_positives1_ix=0

            print("\033[1;31;34m \n","----------------------------")
            print("\033[3;28;88m\n",generator.image_path(i)[5:(len(generator.image_path(i))-4)],": ")
            """
            print("\033[1;31;32m True Positives : \n",true_positives1_i)
            print("\033[1;31;38m False Positives : \n",false_positives1_i)
            print("\033[1;31;35m False Negatives : \n",false_negatives1_i)
            print("\033[1;31;34m \n","----------------------------")
            print("\033[3;28;88m\n",generator.image_path(i)[5:(len(generator.image_path(i))-4)],": ")
            """
            print("\033[1;31;32m True Positives>iou_threshold : \n",true_positives1_ix)
            print("\033[1;31;38m False Positives>iou_threshold : \n",false_positives1_ix)
            print("\033[1;31;35m False Negatives>iou_threshold : \n",false_negatives1_ix)
            TP = true_positives1_ix
            FP = false_positives1_ix
            FN = false_negatives1_ix
            false_positives_i = np.zeros((0,))
            true_positives_i  = np.zeros((0,))
            false_negatives_i = np.zeros((0,))
            false_positives_ix = np.zeros((0,))
            true_positives_ix  = np.zeros((0,))
            false_negatives_ix = np.zeros((0,))
            cv2.imwrite(os.path.join(save_path, '{}.bmp'.format(generator.image_path(i)[5:(len(generator.image_path(i))-4)]+" TP :"+str(TP)+" FP :"+str(FP)+" FN :"+str(FN))), image1) 

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

    # sort by score
        
    indices         = np.argsort(-scores)
    false_positives = false_positives[indices]
    true_positives  = true_positives[indices]
        
    # compute false positives and true positives
    false_positives = np.cumsum(false_positives)
    true_positives  = np.cumsum(true_positives)
    if len(false_negatives) == 0:
        false_negatives1 = 0
    else:
        false_negatives1 = int(max(np.cumsum(false_negatives)))
        false_negatives = np.cumsum(false_negatives)
    try:
        false_positives1 = int(max(false_positives))
    except:
        false_positives1=0
    try:
        true_positives1 = int(max(true_positives))
    except:
        true_positives1=0
    false_positivesx = np.cumsum(false_positivesx)
    true_positivesx  = np.cumsum(true_positivesx)
    if len(false_negativesx) == 0:
        false_negatives1x = 0
    else:
        false_negatives1x = int(max(np.cumsum(false_negativesx)))
        false_negativesx = np.cumsum(false_negativesx)
    try:
        false_positives1x = int(max(false_positivesx))
    except:
        false_positives1x=0
    try:
        true_positives1x = int(max(true_positivesx))
    except:
        pass
    """
    print("\033[03;28;88m \n","----------------------------")
    print("\033[03;01;39m\n","All of Validation Images:")
    print("\033[1;31;32m True Positives : \n",true_positives1)
    print("\033[1;31;38m False Positives : \n",false_positives1)
    print("\033[1;31;35m False Negatives : \n",false_negatives1)
    """
    print("\033[03;28;88m \n","----------------------------")
    print("\033[03;01;39m\n","All of Validation Images:")
    print("\033[1;31;32m True Positives>iou_threshold : \n",true_positives1x)
    print("\033[1;31;38m False Positives>iou_threshold : \n",false_positives1x)
    print("\033[1;31;35m False Negatives>iou_threshold : \n",false_negatives1x)
   
    # compute recall and precision
    recall    = true_positives / num_annotations
    recallx    = max(true_positivesx) / num_annotations
    precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
    precision_allx = max(true_positivesx) / (max(true_positivesx + false_positivesx))
    precision_all = max(true_positives) / (max(true_positives + false_positives))
    # compute average precision
    average_precision  = _compute_ap(recall, precision)
    average_precisions[label] = average_precision, num_annotations
    accuracy = max(true_positives) / (max(true_positives + false_positives + max(false_negatives)))
    accuracyx = max(true_positivesx) / (max(true_positivesx + false_positivesx + max(false_negativesx)))
    print("\033[03;01;39m \n","----------------------------")
    """
    print("\033[03;01;39m \n","Validation accuracy: ")
    print("\033[03;01;39m \n","recall: ",max(recall))
    print("\033[03;31;39m \n","average_precisions: ",average_precisions[label][0])
    print("\033[03;01;39m \n","F1 score: ",(2*((precision_all*max(recall))/(precision_all+max(recall)))))
    print("\033[03;31;39m \n","Accuracy: ",accuracy) 
    print("\033[03;31;39m \n","Precision: ",precision_all)
    print("\033[03;31;34m \n","Number of Annotations: ",average_precisions[label][1])
    """
    print("\033[03;01;39m \n","Validation accuracy: ")
    print("\033[03;31;39m \n","average_precisions: ",average_precisions[label][0])
    print("\033[03;01;39m \n","recall>iou_threshold: ",recallx)
    print("\033[03;31;39m \n","Accuracy>iou_threshold: ",accuracyx) 
    print("\033[03;31;39m \n","Precision>iou_threshold: ",precision_allx)
    print("\033[03;01;39m \n","F1 score>iou_threshold: ",(2*((precision_allx*recallx)/(precision_allx+recallx))))
    print("\033[03;31;34m \n","Number of Annotations: ",average_precisions[label][1])
    print("\033[03;01;39m \n","----------------------------")
    if save_log is not False:
        try:
            os.mkdir('Results/')
        except:
            pass
        f1=2*((precision_allx*recallx)/(precision_allx+recallx))
        p=np.array([int(save_log[:-3]),true_positives1x,false_positives1x,false_negatives1x,np.float(average_precisions[label][0]),np.float(recallx),np.float(accuracyx),np.float(precision_allx),np.float(f1)])
        #name='Results/{}.txt'.format(save_log[:-3])
        #np.savetxt(name,p)
        return(p)
    return output
# Editted by Mohammad Rahimzadeh (mr7495@yahoo.com)
