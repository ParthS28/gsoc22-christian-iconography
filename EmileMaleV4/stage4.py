import numpy as np
import pandas as pd
import os 
import cv2 
import mediapipe as mp
from scipy.special import softmax
from scipy import spatial
np.set_printoptions(precision=5, suppress=True)

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils


images = []
for path in os.listdir('./data/images/'):
# check if current path is a file
    if os.path.isfile(os.path.join('./data/images/', path)):
        images.append(path)

classes = ['baby','person','angel','book','jar','crown','bird','crescent','flowers','crucifict','pear','skull','lamb']
df = pd.DataFrame(columns=['item', 'birth_virgin', 'marriage', 'annunciation', 'birth_jesus', 'adoration', 'coronation', 'assumption', 'death', 'virgin_and_child'])


# 0 - Birth of Virgin Mary/Nativity of the blessed Virgin Mary
# 1 - Marriage of the virgin
# 2 - The annunciation of the Blessed Virgin
# 3 - Nativity of Jesus
# 4 - Adoration of Magi
# 5 - Coronation of the Virgin
# 6 - Assumption of the Virgin
# 7 - Death of the Virgin
# 8 - Virgin and the child
mary_classes = ['birth_virgin', 'marriage', 'annunciation', 'birth_jesus', 'adoration', 'coronation', 'assumption', 'death', 'virgin_and_child']

for image in images:
    scores = np.zeros([9])
    # print(image)
    if not os.path.exists('out2/exp/labels/'+image.split('.')[0]+'.txt'):
        print('out2/exp/labels/'+image.split('.')[0]+'.txt not present')
        scores = softmax(scores)
        scores = scores.tolist()
        scores.insert(0, image)
        continue

    with open('out2/exp/labels/'+image.split('.')[0]+'.txt', 'r') as f:
        lines = f.readlines()

    img = cv2.imread('data/images/'+image)
    dh, dw, dc = img.shape
    present_in_image = []
    coordinates = []
    for line in lines:
        # print(line)
        num = int(line.split(' ')[0])
        present_in_image.append(classes[num])
        class_id, x_center, y_center, w, h, prob = line.strip().split()
        x_center, y_center, w, h = float(x_center), float(y_center), float(w), float(h)
        x_center = round(x_center * dw)
        y_center = round(y_center * dh)
        w = round(w * dw)
        h = round(h * dh)
        x = round(x_center - w / 2)
        y = round(y_center - h / 2)
        coordinates.append([x, y, w, h])
        # imgCrop = img[y:y + h, x:x + w]
        # cv2.imshow("Crop Image",imgCrop)

        cv2.waitKey(0)

    ### Scoring system
    if('person' in present_in_image and 'baby' in present_in_image):
        # add one to all the classes that have person and baby
        scores[0]+=1
        scores[3]+=1
        scores[4]+=1
        scores[8]+=1

    if(len(list(set(present_in_image))) < 4):
        scores[8]+=1
        
    if('crown' in present_in_image):
        scores[5]+=1
        scores[8]+=1

    if('bird' in present_in_image):
        scores[2]+=1
        scores[5]+=1
        scores[8]+=1

    if('lamb' in present_in_image):
        scores[3]+=1
        
    if('flowers' in present_in_image):
        scores[2]+=1
        scores[8]+=1

    if(present_in_image.count('person') > 2):
        scores[0]+=2
        scores[1]+=2
        scores[3]+=2
        scores[4]+=2
        scores[6]+=2
        scores[7]+=2

    if('jar' in present_in_image):
        scores[4]+=1

    if(len(present_in_image) < 4):
        scores[8]+=len(present_in_image)

    if(present_in_image.count('angel') > 0):
        scores[2]+=1
        scores[5]+=1
        scores[6]+=1

    if('book' in present_in_image):
        scores[2]+=1
        

    for i in range(len(coordinates)):
        if(present_in_image[i] == 'person'):
            x,y,w,h = coordinates[i]
            imgCrop = img[y:y + h, x:x + w]
            imgRGB = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)
            if results.pose_world_landmarks:
                # Nose
                nose = [results.pose_world_landmarks.landmark[0].x, results.pose_world_landmarks.landmark[0].y, results.pose_world_landmarks.landmark[0].z]


                # Right hand
                right_hand = [[results.pose_world_landmarks.landmark[16].x, results.pose_world_landmarks.landmark[16].y, results.pose_world_landmarks.landmark[16].z],
                            [results.pose_world_landmarks.landmark[18].x, results.pose_world_landmarks.landmark[18].y, results.pose_world_landmarks.landmark[18].z],
                            [results.pose_world_landmarks.landmark[20].x, results.pose_world_landmarks.landmark[20].y, results.pose_world_landmarks.landmark[20].z],
                            [results.pose_world_landmarks.landmark[22].x, results.pose_world_landmarks.landmark[22].y, results.pose_world_landmarks.landmark[22].z]
                ]


                # Left hand
                left_hand = [[results.pose_world_landmarks.landmark[15].x, results.pose_world_landmarks.landmark[15].y, results.pose_world_landmarks.landmark[15].z],
                            [results.pose_world_landmarks.landmark[17].x, results.pose_world_landmarks.landmark[17].y, results.pose_world_landmarks.landmark[17].z],
                            [results.pose_world_landmarks.landmark[19].x, results.pose_world_landmarks.landmark[19].y, results.pose_world_landmarks.landmark[19].z],
                            [results.pose_world_landmarks.landmark[21].x, results.pose_world_landmarks.landmark[21].y, results.pose_world_landmarks.landmark[21].z]
                ]

                right_shoulder = [results.pose_world_landmarks.landmark[12].x, results.pose_world_landmarks.landmark[12].y, results.pose_world_landmarks.landmark[12].z]

                left_shoulder = [results.pose_world_landmarks.landmark[11].x, results.pose_world_landmarks.landmark[11].y, results.pose_world_landmarks.landmark[11].z]

                avg_rh = np.mean(right_hand, axis = 0)

                avg_lh = np.mean(left_hand, axis = 0)
                r_horizontal_dist = abs(avg_rh[0] - right_shoulder[0])
                r_vertical_dist = abs(avg_rh[1] - right_shoulder[1])
                l_horizontal_dist = abs(avg_lh[0] - left_shoulder[0])
                l_vertical_dist = abs(avg_lh[1] - left_shoulder[1])

                ### if left hand and right hand are close together, most likely prayer
                dist = spatial.distance.cosine(avg_rh, avg_lh)
                if(dist < 0.01):
                    # print('prayer', dist)
                    scores[2]+=1
                    scores[4]+=1
                    # scores[5]+=1
                    scores[6]+=1
                    scores[7]+=1
                elif((r_horizontal_dist > 0.15 and r_vertical_dist < 0.2) and (l_horizontal_dist > 0.15 and l_vertical_dist < 0.2)):
                    # print('both hands extended')
                    scores[6]+=1
                    # scores[0]+=1 #
                elif((r_horizontal_dist > 0.15 and r_vertical_dist > 0.2) or (l_horizontal_dist > 0.15 and l_vertical_dist > 0.2)):
                    # print('extended hand')
                    scores[1]+=1
                    # scores[4]+=1
                    scores[5]+=1
                    # scores[6]+=1
                # lying down - nose and shoulder on a similar height
                elif(abs(nose[1] - ((right_shoulder[1] +left_shoulder[1]) / 2)) < 0.05):
                    # print('lying_down')
                    scores[7]+=1
                    scores[0]+=1 #
                


    # print(image, scores)
    scores = softmax(scores)
    
    scores = scores.tolist()
    
    scores.insert(0, image)
    
    df.loc[df.shape[0]] = scores
df.round(5)
df.to_csv('class_prediction.csv')

