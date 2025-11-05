import edge_tts
import asyncio


from pydub import AudioSegment
from pydub.playback import play


class TextToSpeech:
    def __init__(self):
        self.voice = "zh-CN-XiaoxiaoNeural"  # 中文语音

    async def synthesize_async(self, text, output_path):
        """异步文本转语音"""
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(output_path)

    def synthesize(self, text, output_path="output_audio.mp3"):
        """文本转语音"""
        # 如果使用ChatTTS，在这里替换实现
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.synthesize_async(text, output_path))
        return output_path

    def play(self, audio_path="output_audio.mp3"):
        audio = AudioSegment.from_mp3(audio_path)
        play(audio)


if __name__ == "__main__":
    tts = TextToSpeech()
    tts.synthesize("你好啊，你在干什么啊")
    tts.play()

    # playsound(tts.synthesize("你好"))
