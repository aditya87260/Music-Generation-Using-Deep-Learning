from music21 import converter, instrument, note, chord, stream
import glob
import pickle
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import *

#Read a Midi File
midi = converter.parse("midi_songs/EyesOnMePiano.mid")#converter is used to read any midi file.Here we pass the midi file in the parse function of converter.
#print(midi)  #In copy
#midi.show('midi') # it is used to play the midi music files
#midi.show("text") #it will print the whole music in text format.It will print all the notes and chords in sequence.

# Flat all the elements
elements_to_parse = midi.flat.notes # in any score container chords and notes of a particular music is present but inside a score container there are also some sub containers.So chords and notes are present at these diff containers.We can think of it like a list contains a sub list and inside that also there are sub list containing different notes and chords.So here we have flatten the list so that all the content i.e. all the chords and notes comes in a single list

#print(len(elements_to_parse)) # print no. of element now in that single list/container

#for e in elements_to_parse:
    #print(e, e.offset) #Here we have printed one by one all the elements of container or list e.offset will print the time of a particular note or chord in the music

#now we know that there are only two types of element in the music file that are notes and chords.So in music21 using ptch method we can get the name of a particular note and using normalorder method we can get chord value for a particular element of music file.So now we will store notes andchords individually in notes_demo 
#isinstance method return whether the particular element is note or chord
#isinstance(elements_to_parse[68], chord.Chord)
notes_demo = []

for ele in elements_to_parse:
    # If the element is a Note,  then store it's pitch
    if isinstance(ele, note.Note):
        notes_demo.append(str(ele.pitch))
    
    # If the element is a Chord, split each note of chord and join them with + ex- '1+2+7' denotes  that this chord has three element or notes.Here each numerical value denotes a particular note value
    elif isinstance(ele, chord.Chord):
        notes_demo.append("+".join(str(n) for n in ele.normalOrder)) # stor chord value in string format

#print(len(notes_demo))


#Preprocessing all Files
# now we have preprocessed all the midi music files and we have stored notes and chords of all musics in a notes file 
"""""
notes = []

for file in glob.glob("midi_songs/*.mid"):
    midi = converter.parse(file) # Convert file into stream.Score Object
    
    print("parsing %s"%file)
    
    elements_to_parse = midi.flat.notes
    
    
    for ele in elements_to_parse:
        # If the element is a Note,  then store it's pitch
        if isinstance(ele, note.Note):
            notes.append(str(ele.pitch))

        # If the element is a Chord, split each note of chord and join them with +
        elif isinstance(ele, chord.Chord):
            notes.append("+".join(str(n) for n in ele.normalOrder))

print(len(notes))

with open("notes", 'wb') as filepath:
    pickle.dump(notes, filepath)

"""""

with open("notes", 'rb') as f:
    notes= pickle.load(f)

n_vocab = len(set(notes))

#print("Total notes- ", len(notes))  # so after preprocessing all the songs we have got 60886 notes and chord values
#print("Unique notes- ",  n_vocab)   # we have got 359 unique notes and chords values so it means in our lstm model we will have 359 diff posibilities for a particular music elemnt while prediction because it denotes that we have 359 diff classes

#print(notes[100:200])


#Prepare Sequential Data for LSTM
#in copy
# How many elements LSTM input should consider
sequence_length = 100

# All unique classes
pitchnames = sorted(set(notes))

# Mapping between ele to int value
ele_to_int = dict( (ele, num) for num, ele in enumerate(pitchnames) )

network_input = []
network_output = []

for i in range(len(notes) - sequence_length):
    seq_in = notes[i : i+sequence_length] # contains 100 values
    seq_out = notes[i + sequence_length]

    network_input.append([ele_to_int[ch] for ch in seq_in])
    network_output.append(ele_to_int[seq_out])

# No. of examples
n_patterns = len(network_input)
#print(n_patterns)

# Desired shape for LSTM
network_input = np.reshape(network_input, (n_patterns, sequence_length, 1)) # lstm receives data in 3d form so thats why we have to convert this data in 3d
#print(network_input.shape)

normalised_network_input = network_input/float(n_vocab)  # normalize the data

# Network output are the classes, encode into one hot vector
network_output = np_utils.to_categorical(network_output)  # convert Y-data into one hot vector

#print(network_output.shape)
#print(normalised_network_input.shape)
#print(network_output.shape)
"""""
#Create Model

from keras.models import Sequential, load_model
from keras.layers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping

model = Sequential()
model.add( LSTM(units=512,
               input_shape = (normalised_network_input.shape[1], normalised_network_input.shape[2]),
               return_sequences = True) )
model.add( Dropout(0.3) )
model.add( LSTM(512, return_sequences=True) )
model.add( Dropout(0.3) )
model.add( LSTM(512) )
model.add( Dense(256) )
model.add( Dropout(0.3) )
model.add( Dense(n_vocab, activation="softmax") )

model.compile(loss="categorical_crossentropy", optimizer="adam")
model.summary()

checkpoint = ModelCheckpoint("model.hdf5", monitor='loss', verbose=0, save_best_only=True, mode='min')


model_his = model.fit(normalised_network_input, network_output, epochs=100, batch_size=64, callbacks=[checkpoint])
"""""

model = load_model("new_weights.hdf5")
#Predictions
sequence_length = 100
network_input = []

for i in range(len(notes) - sequence_length):
    seq_in = notes[i : i+sequence_length] # contains 100 values
    network_input.append([ele_to_int[ch] for ch in seq_in])

# Any random start index
start = np.random.randint(len(network_input) - 1)

# Mapping int_to_ele
int_to_ele = dict((num, ele) for num, ele in enumerate(pitchnames))

# Initial pattern 
pattern = network_input[start]
prediction_output = []

# generate 200 elements
for note_index in range(500):
    prediction_input = np.reshape(pattern, (1, len(pattern), 1)) # convert into numpy desired shape 
    prediction_input = prediction_input/float(n_vocab) # normalise
    
    prediction =  model.predict(prediction_input, verbose=0)
    
    idx = np.argmax(prediction)
    result = int_to_ele[idx]
    prediction_output.append(result) 
    
    # Remove the first value, and append the recent value.. 
    # This way input is moving forward step-by-step with time..
    pattern.append(idx)
    pattern = pattern[1:]

print(prediction_output)


#Create Midi File
offset = 0 # Time
output_notes = []

for pattern in prediction_output:
    
    # if the pattern is a chord
    if ('+' in pattern) or pattern.isdigit():
        notes_in_chord = pattern.split('+')
        temp_notes = []
        for current_note in notes_in_chord:
            new_note = note.Note(int(current_note))  # create Note object for each note in the chord
            new_note.storedInstrument = instrument.Piano()
            temp_notes.append(new_note)
            
        
        new_chord = chord.Chord(temp_notes) # creates the chord() from the list of notes
        new_chord.offset = offset
        output_notes.append(new_chord)
        
    
    else:
            # if the pattern is a note
        new_note = note.Note(pattern)
        new_note.offset = offset
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)
        
    offset += 0.5

# create a stream object from the generated notes
midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp = "test_output.mid")

midi_stream.show('midi')