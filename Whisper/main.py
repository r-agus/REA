# La primera ejecución del programa tardará más porque tiene que descargar el modelo

import whisper                              # Hacer comando: pip install openai-whisper
import ffmpeg                               # Hacer comando: pip install ffmpeg-python
import customtkinter as tk                  # Hacer comando: pip install customtkinter
import time

AUDIO_TRANSCRIBIR = "audio.mp3"
SEGUNDOS_ESCUCHA = 5
FICHERO_TRANSCRIPCION = "transcripcion.txt"

# Se puede cambiar el modelo, en orden de peor a mejor es:
# tiny < base < small < medium < large
# la transcripción tarda más también si el modelo es mejor
MODELO = "base"

# Es necesario instalar ffmpeg y poner el siguiente comando en un CLI:
# ffmpeg -f avfoundation -list_devices true -i ""
# cambiar el valor de la constante por el nombre del micrófono a utilizar
MICROFONO = ':MacBook Air (micrófono)'

# Grabo el audio con la librería de ffmpeg para python
def grabar_audio(nombre_archivo, duracion_segundos):
    grabando_label.configure(text="GRABANDO", text_color="red")
    root.update()
    (
        ffmpeg
        .input(MICROFONO, format='avfoundation', t=duracion_segundos)
        .output(nombre_archivo, y='-y')
        .run()
    )
    grabando_label.configure(text="")
    root.update()

# Borrar el texto de la etiqueta
def borrar_feedback():
    feedback_label.configure(text="")
    grabando_label.configure(text="")

def transcribir():
    grabar_audio(AUDIO_TRANSCRIBIR, SEGUNDOS_ESCUCHA)      # Grabación de audio, se especifica el archivo en el que se guarda y los segundos de grabación
    model = whisper.load_model(MODELO)                     # Modelo que se va a utilizar en la transcripción
    result = model.transcribe(AUDIO_TRANSCRIBIR)           # Nombre del archivo a transcribir

    # Abre el archivo en modo escritura ('w')       
    with open(FICHERO_TRANSCRIPCION, 'w') as f:              # Escritura del resultado en el fichero especificado
        # Escribe texto en el archivo
        f.write(result["text"])

    # Mostrar un mensaje de confirmación
    feedback_label.configure(text="¡Transcripción completada!")

    root.after(1000, borrar_feedback)

tk.set_appearance_mode("system")
tk.set_default_color_theme("dark-blue")

root = tk.CTk()
root.geometry("300x300")

frame = tk.CTkFrame(master=root)
frame.pack(fill="both", expand=True)

button = tk.CTkButton(master=frame, text="Grabar", command=transcribir)
button.place(relx=0.5, rely=0.5, anchor="center")

grabando_label = tk.CTkLabel(master=frame, text="")
grabando_label.place(relx=0.5, rely=0.5, anchor="n", y=30)

feedback_label = tk.CTkLabel(master=frame, text="")
feedback_label.place(relx=0.5, rely=0.5, anchor="n", y=60)

root.mainloop()

