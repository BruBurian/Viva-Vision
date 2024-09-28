let trilho = document.getElementById('trilho')
let body = document.querySelector('body')

trilho.addEventListener('click', ()=>{
    trilho.classList.toggle('dark')
    body.classList.toggle('dark')
})

const button = document.getElementById('microfone-button');

button.addEventListener('click', () => {
    navigator.mediaDevices.getUserMedia({ audio: true })
    .then(stream => {
        const mediaRecorder = new MediaRecorder(stream);  

        // Aqui você pode processar o áudio, como gravar em um arquivo ou enviar para um servidor
        mediaRecorder.start();

        // Exemplo de parada da gravação após 5 segundos:
        setTimeout(() => {
            mediaRecorder.stop();
        }, 5000);
    })
    .catch(err => {
        console.error('Erro ao acessar o microfone:', err);
    });
});