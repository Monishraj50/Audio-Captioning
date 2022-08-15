from _future_ import unicode_literals
import youtube_dl
from pydub import AudioSegment
ydl_opts = {
’ format ’: ’ bestaudio / best ’,
#’ username ’: ’ monishraj50@gmail . com ’ ,
#’ password ’: ’ ’ ,
’ postprocessors ’: [{
’key ’: ’ FFmpegExtractAudio ’,
’ preferredcodec ’: ’wav ’,
’ preferredquality ’: ’192 ’ ,
}] ,
}
for i in l:
print (i)
try :
with youtube_dl . YoutubeDL ( ydl_opts ) as ydl :
ydl . download ([ ’ http :// www . youtube . com / watch ?v = ’+ str ( data [’ youtube_id ’][ i ])])
t1 = ( data [’ start_time ’][ i ]) * 1000 # Works in milliseconds
t2 = ( data [’ start_time ’][ i ]+10) * 1000
newAudio = AudioSegment . from_wav ( next (( s for s in os . listdir ( ’./ ’) if data [’ youtube_id ’][ i] in s), None ))
newAudio = newAudio [ t1 : t2 ]
newAudio . export ( ’./ audiocaps_extract / edited_audiocaps / ’+ str ( data [ ’ audiocap_id ’][ i ])+ ’ - ’+ str ( data [’ youtube_id ’][ i ])+ ’. wav ’ , format =" wav ")
os . remove ( next (( s for s in os . listdir ( ’./ ’) if data [’ youtube_id ’][ i] in s) , None ))
except ( youtube_dl . utils . ExtractorError , youtube_dl . utils . DownloadError ):
print (i)
pass
