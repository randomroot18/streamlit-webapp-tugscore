import streamlit as st
# from streamlit_webrtc import (VideoProcessorBase, RTCConfiguration,WebRtcMode,webrtc_streamer)
# from utils import *
# import av
# from PIL import Image
import base64
import requests
import numpy as np
import tempfile
# import cv_pipeline
import cv2
import numpy as np
import os 
from matplotlib import pyplot as plt 
import time 
import mediapipe as mp
import math 
# from tinytag import TinyTag
from scipy.signal import argrelextrema
from scipy.signal import find_peaks
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle 

def render_svg(svg):
    # Renders the given SVG string
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    st.write(html, unsafe_allow_html=True)

def render_png(png):
    # Renders the given PNG data
    b64 = base64.b64encode(png).decode("utf-8")
    html = r'<img src="data:image/png;base64,%s"/>' % b64
    st.write(html, unsafe_allow_html=True)


mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose


person_height=1
# person_weight=float(input("Participant Weight(lbs): "))
# person_age=int(input("Participant Age: "))
dist_from_cam=1 
f_length=0
t_f=False
duration=0.00

frame_num=0
dist_list=[]
#Proportions: trunk = 29.5% of body height


#Body segment weigh data (de Leva's)

head_weight=.06810
trunk_weight=.43020
total_arm_weight=.04715 #SINGLE ARM
total_leg_weight=.20370 #SINGLE LEG

trunk_length=0.295

frames_per_sec=0
total_frames=0
duration=0

time_list=[]
# time_list_keypress=[]

cy_27_list=[]
cy_28_list=[]
cy_23_list=[]
cy_24_list=[]

cx_27_list=[]
cx_28_list=[]
cx_24_list=[]
cx_23_list=[]

cz_27_list=[]
cz_28_list=[]
cz_24_list=[]
cz_23_list=[]

cent_x_list=[]
cent_y_list=[]
cent_z_list=[]

COM_x_list=[]
COM_y_list=[]
COM_z_list=[]

cur_dist=0


def mp_func(img,model_name):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img.flags.writeable =False
    results = model_name.process(img)
    img.flags.writeable =True
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    return img,results

def avg( a, b):
    if len(a)==len(b):
        avg_list=np.zeros(len(a))
        for i in range(0,len(a)):
            avg_list[i]=float((a[i]+b[i])/2)
        return avg_list
    else:
        "dimensions unequal ({len(a)},{len(b)})"
        return None

def avg_int(a,b):
    return((a+b)/2)
    
def dim_change(img,scale_percent=50):
    # returns new dimensions or dim
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return dim

def centroid(*points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    z_coords = [p[2] for p in points]
    _len = len(points)
    centroid_x = sum(x_coords)/_len
    centroid_y = sum(y_coords)/_len
    centroid_z = sum(y_coords)/_len
    return (centroid_x, centroid_y)

def shuffle_detector(coord,temp_coord,val_diff):
    if ((temp_coord-coord)/val_diff >=1) or ((temp_coord-coord)/val_diff <=-1):
        return True
    else:
        return False
    
def focal_distance(img_height):
    global f_length
    global dist_from_cam
    f_length = float((img_height*dist_from_cam)/(person_height*trunk_length))

def current_distance(img_height):
    return float((f_length*person_height*trunk_length)/img_height)

def toggle_cont():
    global t_f
    if t_f==True:
        t_f=False
    else:
        t_f=True

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    return pose

def derivative_list_calc(y_list,t_list):
    new_t_list=np.zeros(len(t_list))
    dy_dt=np.zeros(len(y_list))
    print("length:",len(y_list))
    for i in range(0,len(y_list)-1):
        print("index",i)
        dy=y_list[i+1]-y_list[i]
        dt=t_list[i+1]-t_list[i]
        dy_dt[i]=dy/dt
        new_t_list[i]=t_list[i]
        print(dy_dt[i])
    return dy_dt,new_t_list

def plot_notsubplot(x1,x2,y1,y2,t_list,name_title,cap_name):
    
    plt1=plt.figure(1)
    plt2=plt.figure(2)
    plt3=plt.figure(3)

    plt1.title(str(cap_name)+" "+str(name_title)+" X vs Y")
    plt1.plot(x1,y1,color="red",label="left")
    plt1.plot(x2,y2,color="blue",label="right")
    # plt1.plot(list_avg(x1,x2),list_avg(y1,y2),color="green",label="avg")
    plt1.legend()
    plt1.savefig('opencv-falldetect-test/fall_data_edited_plots/'+str(cap_name)+"_"+str(name_title)+"_X_vs_Y"+'.png',dpi=500)
    plt1.close("all")


    plt2.title(str(cap_name)+" "+str(name_title)+" X vs Time")
    plt2.plot(t_list,x1,color="red",label=str(name_title)+" x_left")
    plt2.plot(t_list,x2,color="blue",label=str(name_title)+" x_right")
    plt2.plot(t_list,list_avg(x1,x2),color="green",label="exCOM_x_time")
    # plt.plot(time_list,list_avg(y_23,y_24),color="yellow",label="avg_y_time")
    plt2.legend()
    plt2.savefig('opencv-falldetect-test/fall_data_edited_plots/'+str(cap_name)+"_"+str(name_title)+"_X_vs_Time"+'.png',dpi=500)

    
    plt3.title(str(cap_name)+" "+str(name_title)+" Y vs Time")
    plt3.plot(t_list,y1,color="red",label=str(name_title)+" y_left")
    plt3.plot(t_list,y2,color="blue",label=str(name_title)+" y_right")
    plt3.plot(t_list,list_avg(y1,y2),color="green",label="exCOM_y_time")
    # plt.plot(time_list,list_avg(y_23,y_24),color="yellow",label="avg_y_time")
    plt3.legend()
    plt3.savefig('opencv-falldetect-test/fall_data_edited_plots/'+str(cap_name)+"_"+str(name_title)+"_Y_vs_Time"+'.png',dpi=500)


def plot_graphs(x1,x2,y1,y2,t_list,name_title,cap_name):

    plt.figure(str(name_title))
    plt1 = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=1)
    plt2 = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1)
    plt3 = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=1)
  
    
    plt1.set_title("X vs Y")
    plt1.plot(x1,y1,color="red",label="left")
    plt1.plot(x2,y2,color="blue",label="right")
    # plt1.plot(list_avg(x1,x2),list_avg(y1,y2),color="green",label="avg")
    plt1.legend()

    plt2.set_title("X vs Time")
    plt2.plot(t_list,x1,color="red",label=str(name_title)+" x_left")
    plt2.plot(t_list,x2,color="blue",label=str(name_title)+" x_right")
    plt2.plot(t_list,list_avg(x1,x2),color="green",label="avg_x_time")
    # plt.plot(time_list,list_avg(y_23,y_24),color="yellow",label="avg_y_time")
    plt2.legend()

    plt3.set_title("Y vs Time")
    plt3.plot(t_list,y1,color="red",label=str(name_title)+" y_left")
    plt3.plot(t_list,y2,color="blue",label=str(name_title)+" y_right")
    plt3.plot(t_list,list_avg(y1,y2),color="green",label="avg_y_time")
    # plt.plot(time_list,list_avg(y_23,y_24),color="yellow",label="avg_y_time")
    plt3.legend()

# temp commented
    # plt.savefig('opencv-falldetect-test/fall_data_edited_plots/'+str(cap_name)+"_"+str(name_title)+'.png',dpi=500)
    # plt.show()
    # plt.pause(1)
    plt.close("all")




###### FOOT STRIKE
change_frames=[]
frames_per_sec = 0
total_frames= 0
def foot_strike(cap_name):
    
    global change_frames
    global frames_per_sec
    global total_frames
    global time_list
    global results
    # global time_list_keypress
    
    global cx_27_list
    global cx_28_list
    global cx_24_list
    global cx_23_list

    global cy_27_list
    global cy_28_list
    global cy_23_list
    global cy_24_list

    global cent_x_list
    global cent_y_list
    global cent_z_list

    global COM_x_list
    global COM_y_list
    global COM_z_list
    global frames_per_sec
    global total_frames
    global duration
    global cur_dist

    temp_bool=False
    check_temp_bool=False
    time_list=[]

    cy_27_list=[]
    cy_28_list=[]
    cy_23_list=[]
    cy_24_list=[]

    cx_27_list=[]
    cx_28_list=[]
    cx_24_list=[]
    cx_23_list=[]

    cz_27_list=[]
    cz_28_list=[]
    cz_24_list=[]
    cz_23_list=[]

    cent_x_list=[]
    cent_y_list=[]
    cent_z_list=[]

    COM_x_list=[]
    COM_y_list=[]
    COM_z_list=[]


    with open('sit_stand.pkl', 'rb') as f:
        model = pickle.load(f)

    temp_bool=True
    # if video_type=="e":
    #     print("\nEDITED")
    #     cap = cv2.VideoCapture("/home/kali/Desktop/opencv_fall_detection_sample/Toronto GAIT Study/Edited-bottom_angle-videos/"+cap_name)

    # elif video_type=="ue":
    #     print("\nUNEDITED")
    #     cap = cv2.VideoCapture("/home/kali/Desktop/opencv_fall_detection_sample/Toronto GAIT Study/Videos/"+cap_name+".mp4")
    # elif video_type=="path":
    cap = cv2.VideoCapture(cap_name)
    
    frames_per_sec = cap.get(cv2.CAP_PROP_FPS)
    total_frames= cap.get(cv2.CAP_PROP_FRAME_COUNT)
    

    global frame_num
    frame_num=0
    start=time.time()
    start_in=0
    
    with mpPose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2,static_image_mode=False, smooth_landmarks=True, enable_segmentation=True,smooth_segmentation=True) as pose:
        while(cap.isOpened()):
                 
            frame_num+=1
            success, img = cap.read()
            if success==False:
                break
            else:
                h, w, c = img.shape
                # c is channel for number of colours, since RGB it returns 3
                cv2.putText(img,"Frame:"+str(int(frame_num)),
                            (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, 
                            (0, 255, 255), 
                            2,
                            cv2.LINE_4)
                # COM Tracker Circle OUTER
                cv2.circle(img,(990,90),70,(255,125,0),2)
                # COM Tracker Circle INNER
                cv2.circle(img,(990,90),35,(255,125,0),2)
                # Vertical Line
                cv2.line(img, (990,20), (990,160), (255,125,0),2) 
                # Horizontal Line
                cv2.line(img, (920,90), (1060,90), (255,125,0),2)
                img,results = mp_func(img,pose)
                if results.pose_landmarks:
                    time_list.append(time.time())
                    mpDraw.draw_landmarks(img, results.pose_landmarks , mpPose.POSE_CONNECTIONS,)
                    # mpDraw.plot_landmarks(results.pose_world_landmarks, mpPose.POSE_CONNECTIONS)
                    try:
                        n_pose = results.pose_landmarks.landmark
                        pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in n_pose]).flatten())

                        X = pd.DataFrame([pose_row])

                        body_pose_class = model.predict(X)[0]
                        
                        if body_pose_class== "Sitting":
                            temp_bool=False

                        elif body_pose_class== "Standing":
                            temp_bool=True
                        
                        if check_temp_bool!=temp_bool:
                            change_frames.append(frame_num)
                            check_temp_bool=temp_bool
                        

                    except:
                        pass
                    if temp_bool==True:
                        start_in=time.time()
                    for id, lm in enumerate(results.pose_landmarks.landmark):
                        if id==11:
                            ncx11, ncy11 = int(lm.x * w), int(lm.y* h)
                            ncz11=lm.z
                        if id==12:
                            ncx12, ncy12 = int(lm.x * w), int(lm.y* h)
                            ncz12=lm.z
                        if id==13:
                            ncx13, ncy13 = int(lm.x * w), int(lm.y* h)
                            ncz13=lm.z
                        if id==14:
                            ncx14, ncy14 = int(lm.x * w), int(lm.y* h)
                            ncz14=lm.z
                        if id==23:

                            ncx23, ncy23 = int(lm.x * w), int(lm.y* h)
                            ncz23=lm.z
                            cy_23_list.append(ncy23)
                            cx_23_list.append(ncx23)

                        if id==24:

                            ncx24, ncy24 = int(lm.x * w), int(lm.y* h)
                            ncz24=lm.z
                            cy_24_list.append(ncy24)
                            cx_24_list.append(ncx24)
                        
                        if id==27:
                            #left ankle
                            ncx27, ncy27 = int(lm.x * w), int(lm.y* h)
                            ncz27=lm.z
                            cy_27_list.append(ncy27)
                            cx_27_list.append(ncx27)
                            
                        if id==28:
                            #right ankle
                            ncx28, ncy28 = int(lm.x * w), int(lm.y* h)
                            ncz28=lm.z
                            cy_28_list.append(ncy28)
                            cx_28_list.append(ncx28)
                        

                            x_t=int(avg_int(ncx23,ncx24))
                            y_t=int(avg_int(ncy23,ncy24))
                            

                            cv2.circle(img,(x_t,y_t),5,(255,125,0),2)
                            
                            cent_x=int(centroid((ncx11,ncy11,ncz11),(ncx12,ncy12,ncz12),(ncx13,ncy13,ncz13),(ncx14,ncy14,ncz14),(ncx23,ncy23,ncz23),(ncx24,ncy24,ncz24),(ncx27,ncy27,ncz27),(ncx28,ncy28,ncz28))[0])
                            cent_y=int(centroid((ncx11,ncy11,ncz11),(ncx12,ncy12,ncz12),(ncx13,ncy13,ncz13),(ncx14,ncy14,ncz14),(ncx23,ncy23,ncz23),(ncx24,ncy24,ncz24),(ncx27,ncy27,ncz27),(ncx28,ncy28,ncz28))[1])
                            cent_z=int(centroid((ncx11,ncy11,ncz11),(ncx12,ncy12,ncz12),(ncx13,ncy13,ncz13),(ncx14,ncy14,ncz14),(ncx23,ncy23,ncz23),(ncx24,ncy24,ncz24),(ncx27,ncy27,ncz27),(ncx28,ncy28,ncz28))[0])
                            cv2.circle(img,(cent_x,cent_y),5,(255,255,0),2)

                            cent_x_list.append(cent_x)
                            cent_y_list.append(cent_y)
                            cent_z_list.append(cent_z)
                            COM_x_list.append(x_t)
                            COM_y_list.append(y_t)
                            # COM_z_list.append(z_t)
                            
                    t_u_x=avg_int(ncx11,ncx12)
                    t_u_y=avg_int(ncy11,ncy12)

                    t_l_x=avg_int(ncx23,ncx24)
                    t_l_y=avg_int(ncy23,ncy24)
                    
                    t_u=[t_u_x,t_u_y]
                    t_l=[t_l_x,t_l_y]
                    if temp_bool==True:

                        temp_bool=False
                        focal_distance(math.dist(t_u,t_l))
                        print("Focal Distance:",f_length)

                    if temp_bool==False:
                        cur_dist=round(current_distance(math.dist(t_u,t_l)),2)
                        dist_list.append(cur_dist)
                        # print(cur_dist)
                        # cv2.putText(img,"Distance: "+str(float(cur_dist))+" meters",(700, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2,cv2.LINE_4)

                    # print("Running:",frame_num)
                else:
                    print("Done:",frame_num)
                    pass

    cap.release()
    cv2.destroyAllWindows()
    
    end=time.time()
    tot_time=end-start  
    time_detection=end-start_in

    print("\ntime: ",tot_time)
    print("\ntime since detection: ",time_detection)

# def get_frame_num(f_num):
#     return f_num

def get_peaks():
    global peaks_right
    global peaks_left

    xCOM=avg(cx_23_list,cx_24_list)
    yCOM=avg(cy_23_list,cy_24_list)
    # for local maxima
    x_right_local_maxima=np.absolute(np.gradient(cy_28_list))
    peaks_right, _ = find_peaks(x_right_local_maxima,distance=20)

    x_left_local_maxima=np.absolute(np.gradient(cy_27_list))
    peaks_left, _ = find_peaks(x_left_local_maxima,distance=20)

    # x_COM_local_maxima=np.absolute(np.gradient(yCOM))
    # peaks_yCOM, _ = find_peaks(yCOM,distance=20)

    # print(len(peaks_left))
    # print(len(peaks_right))

def TUG_for_age(age):
    # return 0 if tug is fine else returns average tug for that age group
    # reference values : https://blog.summit-education.com/wp-content/uploads/Ligotti_Supplements.pdf
    tug_score=round((change_frames[1]-change_frames[0])/frames_per_sec,2)
    if age<=69:
        if tug_score>7.1 and tug_score<=9.0:
            return 0
        else:
            return 8.1
    
    elif age<=79:
        if tug_score>8.2 and tug_score<=10.2:
            return 0
        else:
            return 9.2

    elif age<=99:
        if tug_score>12.7 and tug_score<=10.2:
            return 0
        else:
            return 11.3


# url = "https://github.com/randomroot18/streamlit-webapp-tugscore/blob/main/careyaya_logo-removebg-preview.png"

# r = requests.get(url) 
# png_data = r.content

# render_png(png_data) 

# picture = st.camera_input("Take a picture")

Name = st.text_input('Name')

Age= st.number_input("Age")

person_height=float(st.number_input("Height of participant (in meters)"))

dist_from_cam=float(st.number_input("Distance from camera in beginning (in meters)"))

# Weight=st.number_input("Weight of participant(in lbs)")

uploaded_file = st.file_uploader("Upload file")


# my_bar = st.progress(0)

st.video(uploaded_file)



# nerd = st.checkbox("Show stats for nerds")

if st.button(label="Submit", help="Press to generate results"):
    with st.spinner("Generating Results ..."):
        if uploaded_file!=None:
            

            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            vf=foot_strike(tfile.name)
            tug_score=round((change_frames[1]-change_frames[0])/frames_per_sec,2 )
            get_peaks()
            num_steps=len(peaks_left)+len(peaks_right)
            
            time_list=[x-time_list[0] for x in time_list]
            
            
            st.success('Done!')

            st.write("Results for {}:".format(Name))

            st.write("Total distance walked: ",max(dist_list)-min(dist_list))

            st.write("Estimated total number of steps walked: ",num_steps)

            st.write("Estimated Cadence (steps per minute):",round((num_steps*60)/tug_score))

            st.write("Estimated TUG score: ", tug_score)

            if tug_score>=12:
                st.write("<font color=‘red’>At Risk of Falling!!</font>",unsafe_allow_html=True)
            else:
                st.write("<font color='green'>Not at risk of falling</font>",unsafe_allow_html=True)
            
            if TUG_for_age(Age)==0:
                st.write("<font color='green'>The TUG score is {} and is in the expected range for the age group</font>".format(tug_score),unsafe_allow_html=True)
            else:
                st.write("<font color=‘red’>The TUG score is {} and is above the expected range,average for this age is {} take necesarry precautions!!</font>".format(tug_score,TUG_for_age(Age)),unsafe_allow_html=True)
            

            with st.expander("See more details"):

                st.write("Video Length:",round(total_frames/frames_per_sec,2))
                st.write("Stood up at: ", str(round(change_frames[0]/frames_per_sec,2))+" seconds")
                st.write("Sat down at: ", str(round(change_frames[1]/frames_per_sec,2))+" seconds")

                
                p1=plt.subplot()
                p1.plot(time_list,np.gradient(cy_27_list),color="red",label=str("ankle")+" y_left")
                p1.plot(time_list,np.gradient(cy_28_list),color="blue",label=str("ankle")+" y_right")
                # p1.vlines(x= time_list_keypress, ymin = 0,ymax=40, color = 'black', label = 'axvline - full height')
                p1.plot(np.absolute(time_list)[peaks_right], np.gradient(cy_28_list)[peaks_right], "x")
                p1.plot(np.absolute(time_list)[peaks_left], np.gradient(cy_27_list)[peaks_left], "x")
                p1.grid()
                p1.legend()
                p1.set_title("Vertical movement of ankles")
                p1.set_xlabel('Time(s)')
                p1.set_ylabel('Y-variation (pixels)')
                p1.plot()
                # plt.set(xlabel='Time(s)', ylabel='Y-variation in pixels')
                st.pyplot(plt)


