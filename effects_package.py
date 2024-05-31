from pydub import AudioSegment
from pydub.playback import play

# Beispiel für eine einfache Verzögerung
def simple_delay(audio_segment, delay_ms=500):
    # Erzeugen einer Pause der Länge delay_ms
    silence = AudioSegment.silent(duration=delay_ms)
    # Audio mit Pause davor
    delayed_audio = silence + audio_segment
    return delayed_audio