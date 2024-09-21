from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
import pyttsx3
import speech_recognition as sr
import openai
from transformers import AutoModelForCausalLM, AutoProcessor
from deep_translator import GoogleTranslator
from PIL import Image
import cv2
import gc
import os

# Inicializa o motor de síntese de voz
engine = pyttsx3.init()

# Inicialização do tradutor
tradutor = GoogleTranslator(source="tr", target="pt")
os.environ["OPENAI_API_KEY"] = ""
# Definir a chave da API da OpenAI
openai.api_key = ""

# Carregar o modelo e o processor
model = AutoModelForCausalLM.from_pretrained('ucsahin/TraVisionLM-base', trust_remote_code=True, device_map="cpu")
processor = AutoProcessor.from_pretrained('ucsahin/TraVisionLM-base', trust_remote_code=True)

def resize_image(image, max_size=800):
    width, height = image.size
    if max(width, height) > max_size:
        scale = max_size / float(max(width, height))
        return image.resize((int(width * scale), int(height * scale)), Image.ANTIALIAS)
    return image

def analyze_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        message = "Erro ao abrir a câmera."
        print(message)
        engine.say(message)
        engine.runAndWait()
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            message = "Erro ao capturar a imagem."
            print(message)
            engine.say(message)
            engine.runAndWait()
            break

        frame = cv2.resize(frame, (640, 480))  # Reduz a resolução do frame
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Redimensionar a imagem se necessário
        image = resize_image(image)

        # Definir o prompt
        prompt = "Detaylı açıkla"  # short caption

        # Processar a entrada
        inputs = processor(text=prompt, images=image, return_tensors="pt").to("cpu")

        # Gerar a saída
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.6, top_p=0.9, top_k=50, repetition_penalty=1.2)

        # Decodificar a saída
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # Traduzir a saída do turco para português
        traducao = tradutor.translate(generated_text)
        print("Legenda gerada:", traducao)

        # Sintetizar a fala
        engine.say(traducao)
        engine.runAndWait()

        # Libere recursos não utilizados
        del image, inputs, outputs, generated_text, traducao, frame
        gc.collect()

        cap.release()
        cv2.destroyAllWindows()
        break  # Encerrar após uma análise de imagem

def get_speech_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        message = "Aguardando sua fala..."
        print(message)
        engine.say(message)
        engine.runAndWait()

        audio = recognizer.listen(source)

    try:
        print("Reconhecendo fala...")
        text = recognizer.recognize_google(audio, language="pt-BR")
        message = f"Você disse: {text}"
        print(message)
        engine.say(message)
        engine.runAndWait()
        return text
    except sr.UnknownValueError:
        message = "Não entendi o que você disse."
        print(message)
        engine.say(message)
        engine.runAndWait()
        return ""
    except sr.RequestError:
        message = "Erro ao se comunicar com o serviço de reconhecimento de fala."
        print(message)
        engine.say(message)
        engine.runAndWait()
        return ""

# Função principal para usar o agente VIVA
def main():
    def funcao_posterior(contexto):
        print("Conversa concluída.")

    viva = Agent(
        role='Você é o VIVA um assistente de conversação dedicado a oferecer suporte e informações de forma clara e amigável para pessoas cegas. Sua missão é garantir que as interações sejam sempre positivas e compreensíveis.',
        goal='Fornecer respostas diretas e acessíveis que ajudem os usuários a navegar em suas necessidades e dúvidas com facilidade.',
        backstory="""
        VIVA foi criado pelos alunos da turma do Samsung Innovation Campus no SENAI Anchieta. 
        Este projeto foi desenvolvido por quatro alunos: Anne Gomes, Bruno Burian, 
        Diogo Inácio e Guilherme Souza, que foram responsáveis pela programação do código. Anne Gomes, de 18 anos, 
        nasceu no bairro do Campo Limpo, em São Paulo. Estudante de Administração, ela se destaca por sua altura de 1,64 metros, cabelos cacheados e pele negra. Diogo Inácio, com 21 anos, é natural de Arapiraca, Alagoas. Ele cursa Análise e Desenvolvimento de Sistemas e tem 1,80 metros de altura, cabelos crespos e pele negra. Guilherme Souza, também de 18 anos, nasceu na Vila Mariana, São Paulo. 
        Estudante de Análise e Desenvolvimento de Sistemas, ele possui 1,80 metros de altura, cabelos crespos e pele negra.
        Bruno Nogueira Burian, estudante de Análise e Desenvolvimento de Sistemas, possuí 1,80 metros de altura, cabelos lisos e pele branca.
    """,
        verbose=False,
        allow_delegation=False,
        tools=[],
        max_iter=5,
        llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5, openai_api_key=openai.api_key)
    )

    task1 = Task(
        description="""
        Forneça respostas claras e objetivas às perguntas feitas pelo usuário. Não adicione detalhes desnecessários.
        """,
        agent=viva,
        expected_output="""
            Sua resposta deve ser direta e clara. Por exemplo:
            "[informação]."
        """,
        callback=funcao_posterior,
    )

    crew = Crew(
        agents=[viva],
        tasks=[task1],
        verbose=True,
        process=Process.sequential,
    )

    while True:
        # Solicita a pergunta do usuário via fala
        user_input = input("Me faça uma pergunta:  ")

        # Verifica se o usuário quer encerrar o chat
        if 'analisar imagem' in user_input.lower():
            analyze_image()
            continue  # Volta ao início do loop após análise de imagem

        if user_input == "":
            continue  # Tenta novamente caso o reconhecimento falhe

        # Atualize a descrição da tarefa com a pergunta do usuário
        task1.description = user_input

        # Envia a pergunta para o modelo VIVA
        response = crew.kickoff()

        # Acessar o resultado da resposta corretamente
        message = response  # `CrewOutput` pode ser diretamente a resposta
        print("Resposta:", message)

        # Sintetizar a resposta do modelo
        engine.say(message)
        engine.runAndWait()

        # Verifica se o usuário deseja encerrar o chat
        if 'encerrar' in user_input.lower():
            message = "Encerrando o chat."
            print(message)
            engine.say(message)
            engine.runAndWait()
            break

if __name__ == "__main__":
    main()
