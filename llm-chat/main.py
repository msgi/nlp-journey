from tts import TextToSpeech
from llm import ChatModel
from stt import SpeechToText


# 主应用集成
import gradio as gr
from pydub import AudioSegment


def resample_pydub(input_path, output_path, target_sr=16000):
    """使用pydub进行重采样"""
    # 加载音频文件
    audio = AudioSegment.from_file(input_path)

    # 设置帧率（采样率）和声道
    audio = audio.set_frame_rate(target_sr).set_channels(1)

    # 导出为WAV文件
    audio.export(output_path, format="wav")

    return output_path


class VoiceChatSystem:
    def __init__(self):
        self.stt = SpeechToText()
        self.chat_model = ChatModel()
        self.tts = TextToSpeech()
        self.messages = [
            {
                "role": "system",
                "content": "你是 LLaMA，你都用中文回答我",
            }
        ]
        self.chat_history = []

    def process_voice_input(self, audio_path):
        """处理语音输入的全流程"""
        try:
            # 1. 语音转文本
            # 转一下采样率
            audio_out = "tmp.wav"
            resample_pydub(audio_path, audio_out)

            user_text = self.stt.transcribe(audio_out)

            if not user_text:
                return "识别失败，请重试", None, self.chat_history

            # 2. 生成回复
            self.messages.append({"role": "user", "content": user_text})
            self.messages, bot_response = self.chat_model.generate_response(
                self.messages
            )
            
            self.chat_history.append((user_text, bot_response))

            clean_response = bot_response.replace("*", "")

            # 4. 文本转语音
            audio_output_path = self.tts.synthesize(clean_response)

            return user_text, audio_output_path, self.chat_history

        except Exception as e:
            return f"处理出错: {str(e)}", None, self.chat_history
    
    def process_text_input(self, user_text):
        """处理语音输入的全流程"""
        try:
            # 2. 生成回复
            self.messages.append({"role": "user", "content": user_text})
            self.messages, bot_response = self.chat_model.generate_response(
                self.messages
            )
            
            self.chat_history.append((user_text, bot_response))
            
            clean_response = bot_response.replace("*", "")

            # 4. 文本转语音
            audio_output_path = self.tts.synthesize(clean_response)

            return user_text, audio_output_path, self.chat_history

        except Exception as e:
            return f"处理出错: {str(e)}", None, self.chat_history


def create_gradio_interface():
    """创建Gradio界面"""
    system = VoiceChatSystem()

    with gr.Blocks(title="语音聊天系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎙️ 语音聊天系统 Demo")
        gr.Markdown("支持语音和文本输入，使用Whisper + llama + edge tts")

        with gr.Row():
            with gr.Column(scale=1):
                # 语音输入区域
                audio_input = gr.Audio(
                    sources=["microphone"], type="filepath", label="语音输入"
                )
                audio_btn = gr.Button("发送语音", variant="primary")

                # 文本输入区域
                text_input = gr.Textbox(
                    label="文本输入", placeholder="输入您的问题...", lines=3
                )
                text_btn = gr.Button("发送文本", variant="secondary")
                
                # 清除按钮
                clear_btn = gr.Button("清空历史", variant="stop")

            with gr.Column(scale=2):
                # 显示识别结果
                recognition_text = gr.Textbox(label="识别结果", interactive=False)

                # 音频输出
                audio_output = gr.Audio(label="语音回复", autoplay=True)

                # 聊天历史
                chatbot = gr.Chatbot(label="对话历史", height=400)
                
                

        # 事件绑定
        audio_btn.click(
            fn=system.process_voice_input,
            inputs=[audio_input],
            outputs=[recognition_text, audio_output, chatbot],
        )

        text_btn.click(
            fn=system.process_text_input,
            inputs=[text_input],
            outputs=[audio_output, audio_output, chatbot],
        )

        
        clear_btn.click(
            fn=lambda: ([], []),
            inputs=[],
            outputs=[chatbot, recognition_text]
        )

    return demo


if __name__ == "__main__":
    # 启动应用
    demo = create_gradio_interface()
    demo.launch()
