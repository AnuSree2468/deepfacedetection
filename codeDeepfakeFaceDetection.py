!pip install labelme tensorflow tensorflow-gpu opencv-python matplotlib albumentations

import os, time, uuid, cv2, json, numpy as np, tensorflow as tf
from matplotlib import pyplot as plt
import albumentations as alb
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG16

IMAGES_PATH = os.path.join('data','images')
number_images = 30
cap = cv2.VideoCapture(1)

for imgnum in range(number_images):
    ret, frame = cap.read()
    imgname = os.path.join(IMAGES_PATH,f'{str(uuid.uuid1())}.jpg')
    cv2.imwrite(imgname, frame)
    cv2.imshow('frame', frame)
    time.sleep(0.5)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

!labelme

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

images = tf.data.Dataset.list_files('data\\images\\*.jpg')
def load_image(x): 
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img
images = images.map(load_image)

image_generator = images.batch(4).as_numpy_iterator()
plot_images = image_generator.next()
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, image in enumerate(plot_images): ax[idx].imshow(image)
plt.show()

for folder in ['train','test','val']:
    for file in os.listdir(os.path.join('data',folder,'images')):
        filename = file.split('.')[0]+'.json'
        old = os.path.join('data','labels',filename)
        new = os.path.join('data',folder,'labels',filename)
        if os.path.exists(old): os.replace(old,new)

augment = alb.Compose([
    alb.RandomCrop(450,450),
    alb.HorizontalFlip(p=0.5),
    alb.RandomBrightnessContrast(p=0.2),
    alb.RandomGamma(p=0.2),
    alb.RGBShift(p=0.2),
    alb.VerticalFlip(p=0.5)],
    bbox_params=alb.BboxParams(format='albumentations',label_fields=['class_labels'])
)

for part in ['train','test','val']:
    for image in os.listdir(os.path.join('data',part,'images')):
        img = cv2.imread(os.path.join('data',part,'images',image))
        coords=[0,0,0.00001,0.00001]
        label_path=os.path.join('data',part,'labels',image.split('.')[0]+'.json')
        if os.path.exists(label_path):
            label=json.load(open(label_path))
            pts=label['shapes'][0]['points']
            coords=[pts[0][0],pts[0][1],pts[1][0],pts[1][1]]
            coords=list(np.divide(coords,[640,480,640,480]))
        try:
            for x in range(60):
                aug=augment(image=img,bboxes=[coords],class_labels=['face'])
                cv2.imwrite(f'aug_data/{part}/images/{image.split(".")[0]}.{x}.jpg',aug['image'])
                ann={'image':image}
                if os.path.exists(label_path):
                    ann['bbox']=aug['bboxes'][0] if len(aug['bboxes']) else [0,0,0,0]
                    ann['class']=1 if len(aug['bboxes']) else 0
                else: ann={'bbox':[0,0,0,0],'class':0}
                json.dump(ann,open(f'aug_data/{part}/labels/{image.split(".")[0]}.{x}.json','w'))
        except Exception as e: print(e)

train_images = tf.data.Dataset.list_files('aug_data\\train\\images\\*.jpg',shuffle=False)
train_images = train_images.map(load_image).map(lambda x:tf.image.resize(x,(120,120))).map(lambda x:x/255)

test_images = tf.data.Dataset.list_files('aug_data\\test\\images\\*.jpg',shuffle=False)
test_images = test_images.map(load_image).map(lambda x:tf.image.resize(x,(120,120))).map(lambda x:x/255)

val_images = tf.data.Dataset.list_files('aug_data\\val\\images\\*.jpg',shuffle=False)
val_images = val_images.map(load_image).map(lambda x:tf.image.resize(x,(120,120))).map(lambda x:x/255)

def load_labels(label_path):
    label=json.load(open(label_path.numpy(),'r',encoding='utf-8'))
    return [label['class']],label['bbox']

train_labels=tf.data.Dataset.list_files('aug_data\\train\\labels\\*.json',shuffle=False)
train_labels=train_labels.map(lambda x:tf.py_function(load_labels,[x],[tf.uint8,tf.float16]))

test_labels=tf.data.Dataset.list_files('aug_data\\test\\labels\\*.json',shuffle=False)
test_labels=test_labels.map(lambda x:tf.py_function(load_labels,[x],[tf.uint8,tf.float16]))

val_labels=tf.data.Dataset.list_files('aug_data\\val\\labels\\*.json',shuffle=False)
val_labels=val_labels.map(lambda x:tf.py_function(load_labels,[x],[tf.uint8,tf.float16]))

train=tf.data.Dataset.zip((train_images,train_labels)).shuffle(5000).batch(8).prefetch(4)
test=tf.data.Dataset.zip((test_images,test_labels)).shuffle(1300).batch(8).prefetch(4)
val=tf.data.Dataset.zip((val_images,val_labels)).shuffle(1000).batch(8).prefetch(4)

def build_model(): 
    input=Input((120,120,3))
    vgg=VGG16(include_top=False)(input)
    c=GlobalMaxPooling2D()(vgg)
    c=Dense(2048,activation='relu')(c)
    class_out=Dense(1,activation='sigmoid')(c)
    r=GlobalMaxPooling2D()(vgg)
    r=Dense(2048,activation='relu')(r)
    box=Dense(4,activation='sigmoid')(r)
    return Model(input,[class_out,box])

facetracker=build_model()
sample=train.as_numpy_iterator().next()
opt=tf.keras.optimizers.Adam(0.0001)

def loc_loss(y,h):
    c=tf.reduce_sum(tf.square(y[:,:2]-h[:,:2]))
    h1=y[:,3]-y[:,1];w1=y[:,2]-y[:,0]
    h2=h[:,3]-h[:,1];w2=h[:,2]-h[:,0]
    s=tf.reduce_sum(tf.square(w1-w2)+tf.square(h1-h2))
    return c+s

classloss=tf.keras.losses.BinaryCrossentropy()
regress=loc_loss

class FaceTracker(Model):
    def __init__(self,m): super().__init__();self.m=m
    def train_step(self,b): 
        X,y=b
        with tf.GradientTape() as t:
            c,box=self.m(X,True)
            loss=regress(tf.cast(y[1],tf.float32),box)+0.5*classloss(y[0],c)
        d=t.gradient(loss,self.m.trainable_variables)
        opt.apply_gradients(zip(d,self.m.trainable_variables))
        return {"loss":loss}

ft=FaceTracker(facetracker)
ft.compile(opt,classloss,regress)
ft.fit(train,epochs=10,validation_data=val)

test_batch=test.as_numpy_iterator().next()
pred=facetracker.predict(test_batch[0])

fig,ax=plt.subplots(4,4,figsize=(15,15))
for i in range(4):
    img=test_batch[0][i]
    box=pred[1][i]
    if pred[0][i]>0.5:
        cv2.rectangle(img,
                      tuple(np.multiply(box[:2],[120,120]).astype(int)),
                      tuple(np.multiply(box[2:],[120,120]).astype(int)),
                      (255,0,0),2)
    ax[i//4,i%4].imshow(img)

facetracker.save('facetracker.h5')
facetracker=load_model('facetracker.h5')

cap=cv2.VideoCapture(1)
while cap.isOpened():
    _,frame=cap.read()
    frame=frame[50:500,50:500,:]
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    resized=tf.image.resize(rgb,(120,120))
    yhat=facetracker.predict(np.expand_dims(resized/255,0))
    box=yhat[1][0]
    if yhat[0]>0.5:
        cv2.rectangle(frame,tuple(np.multiply(box[:2],[450,450]).astype(int)),
                      tuple(np.multiply(box[2:],[450,450]).astype(int)),(255,0,0),2)
        cv2.rectangle(frame,tuple(np.add(np.multiply(box[:2],[450,450]).astype(int),[0,-30])),
                      tuple(np.add(np.multiply(box[:2],[450,450]).astype(int),[80,0])),(255,0,0),-1)
        cv2.putText(frame,'face',tuple(np.add(np.multiply(box[:2],[450,450]).astype(int),[0,-5])),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.imshow('DeepFake Detector',frame)
    if cv2.waitKey(1)&0xFF==ord('q'):break

cap.release()
cv2.destroyAllWindows()
