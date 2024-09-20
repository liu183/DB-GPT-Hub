import gradio as gr

def hello(name):
    return f"Hello {name}!"

iface = gr.Interface(fn=hello, inputs="text", outputs="text")
# iface.launch()
iface.launch(share=True)