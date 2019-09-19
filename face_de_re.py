from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.timer import Timer
import numpy as np
import face_recognition
import pickle

torch.set_grad_enabled(False)
net = RetinaFace(phase="test")

CONFIDENCE = 0.50
NMS_THRESHOLD = 0.30
VIZ_THRESHOLD = 0.50

resize = 1

face_data = pickle.loads(open("face_encodings.pickle", "rb").read())

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path):
    pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


net = load_model(net, "Final_Retinaface.pth")
net.eval()
cudnn.benchmark = True
device = torch.device("cpu")
net = net.to(device)

def face_detector(frame):
    img_raw = frame.copy()
    img = np.float32(img_raw)
    if resize != 1:
        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    loc, conf, landms = net(img)  # forward pass
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

        # ignore low scores
    inds = np.where(scores > CONFIDENCE)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

        # keep top-K before NMS
    order = scores.argsort()[::-1][:5000]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

        # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, NMS_THRESHOLD)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

        # keep top-K faster NMS
    dets = dets[:750, :]
    landms = landms[:750, :]

    dets = np.concatenate((dets, landms), axis=1)

    bboxs = []
    for b in dets:
        if b[4] < VIZ_THRESHOLD:
            continue
        b = list(map(int, b))
        
        margin = 10

        x1,y1,x2,y2 = b[0], b[1], b[2], b[3]
        
        img_h, img_w, _ = frame.shape
        w = x2-x1
        h = y2-y1
        margin = int(min(w,h) * margin / 100)
        x_a = x1 - margin
        y_a = y1 - margin
        x_b = x1 + w + margin
        y_b = y1 + h + margin
        if x_a < 0:
            x_b = min(x_b - x_a, img_w-1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h-1)
            y_a = 0
        if x_b > img_w:
            x_a = max(x_a - (x_b - img_w), 0)
            x_b = img_w
        if y_b > img_h:
            y_a = max(y_a - (y_b - img_h), 0)
            y_b = img_h

        name = ""
        print(name)
        face = frame[y_a:y_b,x_a:x_b]
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb,[(y_a,x_b,y_b,x_a)])
        matches = face_recognition.compare_faces(face_data["encodings"],encodings[0],tolerance=0.55)
        print(matches)
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
 
            for i in matchedIdxs:
                name = face_data["names"][i]
                counts[name] = counts.get(name, 0) + 1
 
            name = max(counts, key=counts.get)
            print("name1" ,name)
        cv2.putText(img_raw, name, (x_a+10, y_a), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.rectangle(img_raw, (x_a, y_a), (x_b, y_b), (255, 0, 0), 1)
        bboxs.append([x_a,y_a,x_b,y_b])
        
    return img_raw,bboxs


cap = cv2.VideoCapture(0)

hasFrame, frame = cap.read()

#vid_writer = cv2.VideoWriter('output-game-u-{}.avi'.format(str(args["video"]).split(".")[0]),cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame.shape[1],frame.shape[0]))

while(1):
    hasFrame, frame = cap.read()
    if not hasFrame:
        continue
    
    outOpencvDnn, bbox = face_detector(frame)
    cv2.imshow("GAME", outOpencvDnn)

    #vid_writer.write(outOpencvDnn)

    key = cv2.waitKey(1) & 0xFF 
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vid_writer.release()