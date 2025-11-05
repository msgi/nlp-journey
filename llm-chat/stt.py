from pywhispercpp.model import Model
import config

whisper_model = config.WHISPER_MODEL_PATH


class SpeechToText:
    def __init__(self, model_path=whisper_model):
        self.model = Model(model_path, redirect_whispercpp_logs_to=None)

    def transcribe(self, audio_path):
        """将音频文件转写为文本"""
        segments = self.model.transcribe(audio_path, language="zh")

        text = " ".join([segment.text for segment in segments])
        return text


if __name__ == "__main__":
    stt = SpeechToText()
    stt.transcribe("output_audio.mp3")
