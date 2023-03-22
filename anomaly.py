from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from tqdm import tqdm
import matplotlib.pyplot as plt

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

#New added by Roman
#from keras.utils import np_utils
#from skimage import io
from sklearn.metrics import confusion_matrix
from sklearn.metrics import *
#from terminaltables import AsciiTable
import random
import seaborn as sns

# Set this so that we can redo experiments.
#np.random.seed(376483)
#untill

import tensorflow as tf
# tf.reset_default_graph()
tf.compat.v1.get_default_graph()
print(tf.compat.v1.get_default_graph())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tflearn.layers.conv import conv_2d, max_pool_2d
import pickle

TRAIN_DIR = './dataset1/Train'
TEST_DIR = './dataset1/Test'

IMG_SIZE = 120
LR = 1e-3
n_epoch = 20


def label_img(img):
    word_label = img.split('_')[0]
    if word_label == 'Anomaly':
        return [1, 0]  # one hot encoding
    elif word_label == 'Normal':
        return [0, 1]  # one hot encoding

def create_train_data():
    train_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        ############################################################
        #    This part is different from sentdex's tutorial
        # Chose to use PIL instead of cv2 for image pre-processing
        ############################################################

        img = Image.open(path)  # Read image syntax with PIL Library
        img = img.convert('L')  # Grayscale conversion with PIL library
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)  # Resizing image syntax with PIL Library

        ############################################################

        train_data.append([np.array(img), np.array(label)])
    shuffle(train_data)
    np.save('train_data.npy', train_data)  # .npy extension = numpy file
    return train_data

train_data = create_train_data()
plt.imshow(train_data[10][0], cmap='gist_gray')
print(train_data[10][1])

def process_test_data():
    test_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        if "DS_Store" not in path:
            img_num = img.split('_')[1]  # images are formatted 'Anomaly_1', 'Normal_2'..

            # PIL LIBRARY instead of cv2
            img = Image.open(path)
            img = img.convert('L')
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)

            test_data.append([np.array(img), img_num])
    shuffle(test_data)
    np.save('test_data.npy', test_data)
    return test_data

####### Define CNN (with layers)
# tf.compat.v1.get_default_graph()
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')  # output
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets',metric='Accuracy')

model = tflearn.DNN(convnet, tensorboard_verbose=3)
# tensorboard_dir='log',
train = train_data[-30000:]
test = train_data[:-30000]

##Data preprocessing

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_y = [i[1] for i in test]

# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# callbacks = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit({'input': X}, {'targets': Y}, n_epoch=2, batch_size=64, validation_set=({'input': test_x}, {'targets': test_y}),
                    snapshot_step=500, show_metric=True, run_id='Anomaly_Normal')

test_data = process_test_data()
fig = plt.figure()
model.save('models/model.text')
for num, data in enumerate(test_data[:12]):
    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(3, 4, num + 1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([data])[0]
    print(model_out)
    if np.argmax(model_out) == 1:
        str_label = 'Anomaly'
    else:
        str_label = 'Normal'

    y.imshow(orig, cmap='OrRd')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
    plt.savefig("Result/classification.png")
plt.show()

#New code
preds=np.round(model.predict(test_x), 0)
print('round test labels', preds)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('\nAccuracy: {:.2f}\n'.format(accuracy_score(test_y, preds)))
print('Precision: {:.2f}'.format(precision_score(test_y, preds, average='micro')))
print('Recall: {:.2f}'.format(recall_score(test_y, preds, average='micro')))
print('F1-score: {:.2f}\n'.format(f1_score(test_y, preds, average='micro')))

from sklearn.metrics import classification_report
print('\nClassification Report\n')
print(classification_report(test_y, preds, target_names=['Anomaly', 'Normal']))

# fit model
clf = OneVsRestClassifier(LogisticRegression())
clf.fit(X, Y)
pred = clf.predict(test_x)
pred_prob = clf.predict_proba(test_x)

# roc curve for classes
fpr = {}
tpr = {}
thresh = {}

n_class = 2

for i in range(n_class):
    fpr[i], tpr[i], thresh[i] = roc_curve(test_y, pred_prob[:, i], pos_label=i)

# plotting
plt.plot(fpr[0], tpr[0], linestyle='--', color='orange', label='Class 0 vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='--', color='green', label='Class 1 vs Rest')
#plt.plot(fpr[2], tpr[2], linestyle='--', color='blue', label='Class 2 vs Rest')
plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('Multiclass ROC', dpi=300)
