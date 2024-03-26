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

class SpeechRecognition:
    grabando_label = None
    root = None
    feedback_label = None

    # Grabo el audio con la librería de ffmpeg para python
    def grabar_audio(self, nombre_archivo, duracion_segundos):
        (
            ffmpeg
            .input(MICROFONO, format='avfoundation', t=duracion_segundos)
            .output(nombre_archivo, y='-y')
            .run()
        )
    
    def grabar_audio_interfaz(self, nombre_archivo, duracion_segundos):
        self.grabando_label.configure(text="GRABANDO", text_color="red")
        self.root.update()
        (
            ffmpeg
            .input(MICROFONO, format='avfoundation', t=duracion_segundos)
            .output(nombre_archivo, y='-y')
            .run()
        )
        self.grabando_label.configure(text="")
        self.root.update()

    # Borrar el texto de la etiqueta
    def borrar_feedback(self):
        self.feedback_label.configure(text="")

    def transcribir(self, audio_transcribir, fichero_transcripcion):
        model = whisper.load_model(MODELO)                     # Modelo que se va a utilizar en la transcripción
        result = model.transcribe(audio_transcribir)           # Nombre del archivo a transcribir

        # Abre el archivo en modo escritura ('w')       
        with open(fichero_transcripcion, 'w') as f:              # Escritura del resultado en el fichero especificado
            # Escribe texto en el archivo
            f.write(result["text"])

    def transcribir_interfaz(self):
        self.grabar_audio_interfaz(AUDIO_TRANSCRIBIR, SEGUNDOS_ESCUCHA)      # Grabación de audio, se especifica el archivo en el que se guarda y los segundos de grabación
        model = whisper.load_model(MODELO)                     # Modelo que se va a utilizar en la transcripción
        result = model.transcribe(AUDIO_TRANSCRIBIR)           # Nombre del archivo a transcribir

        # Abre el archivo en modo escritura ('w')       
        with open(FICHERO_TRANSCRIPCION, 'w') as f:              # Escritura del resultado en el fichero especificado
            # Escribe texto en el archivo
            f.write(result["text"])

        # Mostrar un mensaje de confirmación
        self.feedback_label.configure(text="¡Transcripción completada!")

        self.root.after(1000, self.borrar_feedback)

    def crear_interfaz(self):
        tk.set_appearance_mode("system")
        tk.set_default_color_theme("dark-blue")

        self.root = tk.CTk()
        self.root.geometry("300x300")

        frame = tk.CTkFrame(master=self.root)
        frame.pack(fill="both", expand=True)

        button = tk.CTkButton(master=frame, text="Grabar", command=self.transcribir_interfaz)
        button.place(relx=0.5, rely=0.5, anchor="center")

        self.grabando_label = tk.CTkLabel(master=frame, text="")
        self.grabando_label.place(relx=0.5, rely=0.5, anchor="n", y=30)

        self.feedback_label = tk.CTkLabel(master=frame, text="")
        self.feedback_label.place(relx=0.5, rely=0.5, anchor="n", y=60)

        self.root.mainloop()