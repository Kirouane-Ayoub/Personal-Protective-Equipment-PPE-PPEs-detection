from ultralytics import YOLO
import cv2
import cvzone
import math
import streamlit as st

with st.sidebar : 
    st.image("icon.png")
class_names = [ "Helmet", "No-Helmet", "No-vest", "Person", "Vest"]
tab0 , tab1 = st.tabs(["Home" , "Detection"])
with tab0 : 
    st.header("Here are a few use cases for this project:")
    st.write("""
Construction Site Safety Monitoring: The Worker-Safety computer vision model can be employed to monitor construction sites for compliance with safety regulations. By identifying the presence or absence of helmets, vests, and other personal protective equipment, the model can provide real-time alerts to supervisors, who can then address any safety concerns promptly.

Industrial Workplace Safety Inspections: The model can be used to automate safety inspections in industrial environments, such as factories and manufacturing plants. By detecting workers without helmets, vests, or appropriate safety gear, the model can help reduce the potential for accidents and ensure worker safety.

Insurance Risk Assessment: The Worker-Safety computer vision model can be beneficial for insurance companies when assessing the risk associated with insuring a particular workspace or construction project. By analyzing images, the model can provide a safety score or compliance percentage, which can inform the insurer's decision-making process.

Augmented Reality Training: The model can be integrated into augmented reality (AR) applications to provide real-time feedback and training for workers in potentially hazardous environments. By identifying non-compliant situations, the AR application can offer immediate guidance to the worker on the proper use of safety equipment and proper procedures.

Occupational Health and Safety Compliance Auditing: Governmental and regulatory agencies can utilize the Worker-Safety computer vision model to audit organizations for compliance with occupational health and safety regulations. By analyzing video footage of workplaces or construction sites, the model can identify potential safety violations, which can then be investigated by the relevant authorities.""")
with tab1 : 
    from_ = st.selectbox("Select Detection type : " , 
                         ["file" , "live"])
    if from_ == "file" : 
        source = st.file_uploader("Slect your file : " , 
                                  type=("mp4" , "mkv"))
        if source : 
            source = source.name
    elif from_ == "live" : 
        source = st.text_input("Entre your URL : ")
    
    device = st.sidebar.selectbox("Select your Device :" ,
                                   ["cpu" , "cuda"])
    th = st.sidebar.slider("Select Your accuracy threshold : " , 
                           min_value=0.1 , max_value=1.0)
    if st.button("Click to Start") :
        fpsReader = cvzone.FPS()
        cap = cv2.VideoCapture(source)
        model = YOLO("construction-safety-gsnvb.pt")
        model.to(device)

        frame_window = st.image( [] )
        while True : 
            _ , img = cap.read()
            fps, img = fpsReader.update(img,pos=(50,80),
                                        color=(0,0,255),
                                        scale=2,thickness=3)
    
            results = model(img)
            for result in results : 
                bboxs = result.boxes 
                for box in bboxs : 
                    x1  , y1 , x2 , y2 = box.xyxy[0]
                    x1  , y1 , x2 , y2 = int(x1)  , int(y1) , int(x2) , int(y2)
                    conf = math.ceil((box.conf[0] * 100 ))
                    if box.conf[0] >= th : 

                        clsn = int(box.cls[0])
                        if clsn == 1 or clsn == 2:
                            my_color = (0,0,255)
                        else :
                            my_color = (255, 0, 0)

                        w,h = x2 - x1 , y2 - y1
                    
                        w , h = int(w) , int(h)
                    
                        cvzone.cornerRect(img , (x1 , y1 , w , h) ,
                                           l=7 ,colorR=my_color)
                
                        (wt, ht), _ = cv2.getTextSize(f"{conf} % {class_names[clsn]}", 
                                                      cv2.FONT_HERSHEY_PLAIN,1.5, 2)
                        
                        cv2.rectangle(img, (x1, y1 - 25), 
                                      (x1 + wt, y1), 
                                      (0, 0, 255),
                                      cv2.FILLED)
                        
                        cv2.putText(img, f"{conf} % {class_names[clsn]}", (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_PLAIN,1.5,
                                    (255, 255, 255), 2)

            frame  = cv2.cvtColor( img , cv2.COLOR_BGR2RGB)
            frame_window.image(frame)
