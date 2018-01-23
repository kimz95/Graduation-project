import wfdb
import matplotlib.pyplot as plt
import math
import numpy as np
from numpy import array
import tensorflow as tff
tff.flags.DEFINE_integer("batch_size", 64, "Batch size during training")
tff.flags.DEFINE_integer("eval_batch_size", 8, "Batch size during evaluation")
tff.flags.DEFINE_integer("test_batch_size", 8, "Batch size for testing")
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tff.add(tff.matmul(x, weights['h1']), biases['b1'])
    # Output fully connected layer with a neuron for each class
    out_layer = tff.matmul(layer_1, weights['out']) + biases['out']
    return out_layer
#not use
def NumbersWithinRange(items, lower, upper):
    start = items.bisect(lower)
    end = items.bisect_right(upper)
    return items[start:end]

#BEATS EXTRACTION (list of beats)
def peaks(x, peak_indices, fs, title, figsize=(20, 10), saveto=None):
    #heart rate
    hrs = wfdb.processing.compute_hr(siglen=x.shape[0], peak_indices=peak_indices, fs=fs)
    fig, ax_left = plt.subplots(figsize=figsize)

    ax_left.plot(x, color='#3979f0', label='Signal')
    ax_left.plot(peak_indices, x[peak_indices], 'rx', marker='x', color='#8b0000', label='Peak', markersize=10)

    ax_left.set_title(title)
    ax_left.set_xlabel('Time (ms)')
    ax_left.set_ylabel('ECG (mV)', color='#3979f0')
    ax_left.tick_params('y', colors='#3979f0')
    #plt.show()
    return hrs


# Single beats extraction for record
def extract_beats(record, R_indices, peaks_hr, freq):
    flat_record = [item for sublist in record for item in sublist]
    ecg_beats = [];

    # Extract beats of different lengths depending on heart rate
    for i in range(0, len(R_indices)):
        hr = peaks_hr[R_indices[i]].item();
        if (math.isnan(hr)):
            hr = 70.0;
        samples_per_beat = int(freq * (60.0 / hr))
        start = R_indices[i] - samples_per_beat / 2;
        if (start < 0):
            start = 0;
        end = start + samples_per_beat;
        ecg_beats.append(np.array(flat_record[int(start):int(end)]));

    # Resample beats to get fixed input size for classification
    ecg_nparray = np.empty((len(ecg_beats), 250));
    for i in range(0, len(ecg_beats)):
        ecg_beats[i], _ = wfdb.processing.resample_sig(x=ecg_beats[i], fs=len(ecg_beats[i]), fs_target=250);
        ecg_nparray[i] = ecg_beats[i];

    return ecg_nparray;


def read_record(rec, t0=0, tf=30000):
    # Load the wfdb record and the physical samples
    record = wfdb.rdsamp('/Users/salmasamer/Desktop/untitledfolder/' + rec, sampfrom=t0, sampto=tf, channels=[0])
    annotation = wfdb.rdann('/Users/salmasamer/Desktop/untitledfolder/' + rec, 'atr', sampfrom=t0, sampto=tf, summarize_labels=True)
    freq = record.fs
    sig = record.p_signals
    sig = wfdb.processing.normalize(x=sig, lb=0.0, ub=1.0)

    for idx, val in enumerate(sig):
        record.p_signals[idx, 0] = val

    peak_indices = annotation.sample;
    peaks_hr = peaks(x=record.p_signals, peak_indices=peak_indices, fs=record.fs, title="GQRS peaks on record " + rec);
    ecg_beats = extract_beats(record.p_signals, peak_indices, peaks_hr, freq);

    return ecg_beats, annotation;

#sorted_peak_indices = sorted(peak_indices)
#print('Corrected gqrs detected peak indices:', sorted_peak_indices)

"""for i in range(20):
    print(len(beats[i]))"""

#print(annotation.sample)
"""inputs = []
for q in range(1):

    #for i in range(len(sorted_peak_indices)):
    for i in range(4):
        if i == 0:
            continue
        else:

            temprec = wfdb.rdsamp('/Users/salmasamer/Desktop/untitledfolder/101',sorted_peak_indices[i-1],sorted_peak_indices[i], channels=[0])
            #print("SAMPLING FROM " + str(sorted_peak_indices[i-1]) + " to " + str(sorted_peak_indices[i]))

            sig, fields = wfdb.srdsamp('/Users/salmasamer/Desktop/untitledfolder/101', channels = [0], sampfrom = sorted_peak_indices[i-1],sampto=sorted_peak_indices[i])
            #print (sig == temprec.p_signals)
            if( sig == temprec.p_signals):
                print("HELL YEAH")
            #print(fields['fs'])
            ann = wfdb.rdann('/Users/salmasamer/Desktop/untitledfolder/101', 'atr',sampfrom=sorted_peak_indices[i-1], sampto=sorted_peak_indices[i],summarize_labels=True)
            print("From " + str(sorted_peak_indices[i-1]) + " to " + str(sorted_peak_indices[i]))
            #print(len(sig))
            #print(ann.sample[sorted_peak_indices[i-1] : sorted_peak_indices[i]])
            #print(ann.symbol[sorted_peak_indices[i-1] : sorted_peak_indices[i]])
            #print(ann.sample)
            #print(ann.symbol)
            #start = np.searchsorted(ann.sample, sorted_peak_indices[i-1], 'left')
            #end = np.searchsorted(ann.sample, sorted_peak_indices[i], 'right')
            #rng = np.arange(start, end)
            #print(rng)
            #z = np.where(np.logical_and(ann.sample >= sorted_peak_indices[i-1], ann.sample <= sorted_peak_indices[i]))
            #z = NumbersWithinRange(ann.sample,sorted_peak_indices[i-1],sorted_peak_indices[i])
            #x, fields = wfdb.processing.resample_sig(sig[:,0],fs=fields['fs'], fs_target=200)
            x, fields = wfdb.processing.resample_singlechan(sig[:,0], ann = annotation, fs=fields['fs'], fs_target=200)
            #print(fields.symbol[sorted_peak_indices[i-1]:sorted_peak_indices[i]])
            #inputs.append(x)
"""

t0=0;
tf=600000;
beats = [];
beats, annotation = read_record("105", t0, tf)
print(np.shape(annotation.symbol)) #2D array shape every single *6 for annotation, 2nd *250
print(np.shape(beats))
blobinp = beats[0][: , np.newaxis, np.newaxis]
#print(blobinp.shape)

X = beats;
N = len(beats);
Y = np.empty(N, dtype=int); #for annot
for i in range(0,N):
    Y[i] = {
        'N': 0,
        'V': 1,
        'Q': 2,
        '+': 3,
        '~': 4,
        '|': 5,
        }[annotation.symbol[i]]

Z = Y
print(N)
print(Z)
Y = np.full([len(Z),6],0) #array ken(z=6) and all zeros
print(np.shape(Y))
for i in range(len(Z)):
    Y[i][Z[i]] = 1   #puts 1 in right place

print("Y is " + str(type(Y)) )
print("X is " + str(type(X)) )
print(Y)
learning_rate = 0.5
n_hidden_1 = 256 #nodes, could be changed
num_input = 250
num_classes = 6
num_steps = 10000
display_step = 100
X1 = tff.placeholder(tff.float32,[None, num_input])  #works as inputs  same num of inputs
Y1 = tff.placeholder(tff.float32,[None, num_classes]) #works as outputs  same num of outputs (/classes)

weights = { #between inputs and hidden layer
    'h1': tff.Variable(tff.random_normal([num_input, n_hidden_1])),
    'out': tff.Variable(tff.random_normal([n_hidden_1, num_classes])) #btw hidden and outputs
}
biases = {
    'b1': tff.Variable(tff.random_normal([n_hidden_1])),
    'out': tff.Variable(tff.random_normal([num_classes]))
}
logits = neural_net(X1)
prediction = tff.nn.softmax(logits)

# Define loss and optimizer
loss_op = tff.reduce_mean(tff.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y1))
optimizer = tff.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tff.equal(tff.argmax(prediction, 1), tff.argmax(Y, 1))
accuracy = tff.reduce_mean(tff.cast(correct_pred, tff.float32))

# Initialize the variables (i.e. assign their default value)
init = tff.global_variables_initializer()

learn_X = array(X[0:(int(0.75 * len(X)))])
print(np.shape(learn_X))
learn_Y = array(Y[0:int(0.75 * len(Y))])
print(np.shape(learn_Y))
test_X = array(X[int(0.75 * len(X)):])
print(np.shape(test_X))
test_Y = array(Y[int(0.75 * len(Y)):])
print(np.shape(test_Y))


"""learn_X = array(X[0:(int(0.75 * len(X)))]).reshape(250,int(0.75*len(X)))
learn_Y = array(Y[0:int(0.75 * len(Y))]).reshape(250,int(0.75*len(Y)))
test_X = array(X[int(0.75 * len(X)):]).reshape(250,int(0.75*len(X)))
test_Y = array(Y[int(0.75 * len(X)):]).reshape(250,int(0.75*len(X)))"""

with tff.Session() as sess:

    # Run the initializer, Train
    sess.run(init)

    for step in range(1, len(learn_X)):
        batch_x = learn_X[step-1] #take 1 beat from x and y
        batch_x = array(batch_x).reshape(1,250) #takes normal length (here 250 not 200)
        #print(batch_x)
        #print(np.shape(batch_x))
        #print(np.shape(batch_x))
        batch_y = learn_Y[step-1]
        batch_y = array(batch_y).reshape(1,6)

        #print(batch_y)
        #print(np.shape(batch_y))
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X1: batch_x, Y1: batch_y})  #placeholder batch x as input, y as output 1 by 1
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X1: batch_x,
                                                                 Y1: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")
    #TEST
    total_test_accuracy = 0
    print(test_X.shape[0])
    for i in range(test_X.shape[0]):
        total_test_accuracy += sess.run(accuracy, feed_dict={X1: test_X[i].reshape(1, 250),
                                      Y1: test_Y[i].reshape(1, 6)})

    print("AVERAGE ACCURACY")
    print(total_test_accuracy/test_X.shape[0])
#print(beats)
