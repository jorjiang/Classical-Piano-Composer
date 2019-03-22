""" This module prepares midi file data and feeds it to the neural
    network for training """
import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from fractions import Fraction
from sklearn.preprocessing import MultiLabelBinarizer


def train_network():
    """ Train a Neural Network to generate music """
    notes = get_notes()
    
    network_input, network_output = prepare_sequences(notes)

    model = create_network(network_input)

    train(model, network_input, network_output)

def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
#     parsed_notes = []

#     for file in np.random.choice(glob.glob("/code/dl/Classical-Piano-Composer/midi_songs_bc/*.mid"), 3):
#         midi_data = converter.parse(file)

#         print("Parsing %s" % file)

#         ss = midi_data.flat

#         notes = ss.getElementsByClass(['Chord', 'Note'])
#         parsed_notes.extend(notes)
        
#     off_sets = [note.offset for note in parsed_notes]

#     note_dict = {}

#     for note in parsed_notes:
#         offset = note.offset
#         if isinstance(offset, Fraction):
#                 offset = np.round(float(offset),2)
#         if note.offset in note_dict:
#             if not note.isChord:
#                 note_dict[offset].append(note.nameWithOctave.replace('-', ''))
#             else:
#                 note_dict[offset].extend([p.nameWithOctave.replace('-', '') for p in note.pitches])
#         else:
#             if not note.isChord:
#                 note_dict[offset] = [note.nameWithOctave.replace('-', '')]
#             else:
#                 note_dict[offset] = [p.nameWithOctave.replace('-', '') for p in note.pitches]


#     for offset in note_dict:
#         note_dict[offset] = list(set(note_dict[offset]))

#     values = [v for v in note_dict.values()]
    
#     mb = MultiLabelBinarizer()
#     all_values_binary = mb.fit_transform(values)
#     training_data = {'data': all_values_binary, 'binarizer': mb}
#     print(all_values_binary.shape)
#     with open('data/training_data_bach', 'wb') as filepath:
#         pickle.dump(training_data, filepath)
    with open('./data/training_data_bach', 'rb') as filepath:
        all_values_binary = pickle.load(filepath)['data']
    return all_values_binary

def prepare_sequences(notes):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 32
    network_input = []
    network_output = []

    for i in range(0, notes.shape[0] - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append(sequence_in)
        network_output.append(sequence_out)

    network_input = np.array(network_input)
    network_output = np.array(network_output)

    return (network_input, network_output)

def create_network(network_input):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.2))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.2))
    model.add(Dense(network_input.shape[2]))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.load_weights('weights-improvement-30-1.5473-bigger.hdf5')
    return model

def train(model, network_input, network_output):
    """ train the neural network """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=400, batch_size=32, callbacks=callbacks_list)

if __name__ == '__main__':
    train_network()
