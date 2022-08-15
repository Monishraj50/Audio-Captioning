def extract_spect_files ( data , length , save = None ):
X = []
y = []
count =0
for path , category in zip ( data . wav , data . captions ):
audio , sr = librosa . load ( path )
if len ( audio ) < sr * length :
if sr * length > len ( audio ):
max_offset = sr * length - len ( audio )
offset = np . random . randint ( max_offset )
else :
offset = 0
audio = np . pad ( audio , ( offset , sr * length - len ( audio ) - offset ), " constant ")
wav_features = spect_extract ( audio , sr )
X. append ( wav_features )
y. append ( category )
count +=1
printProgressBar ( count , data . shape [0])
if save :
np . savez ( save , features =X , labels =y)
return X ,y
import sys
def printProgressBar (i , max ):
n_bar =50 # size of progress bar
j= i/ max
sys . stdout . write ( ’\r ’)
sys . stdout . write (f "|{ ’ ’ * int ( n_bar * j ):{ n_bar }s }| { int (100 * j )}%")
sys . stdout . flush ()
