# Implantação da solução

## Hospedagem

A API utilizada foi hospedada na plataforma Render, utilizando o código diretamente do GitHub. Render é uma plataforma que possibilita a hospedagem gratuita de aplicações e oferece ferramentas para que essas aplicações sejam executadas sem problemas. O Render atende a proposta da aplicação, porque a mesma não vai ser uma aplicação com um grande tráfego de usuários, logo os impactos de perfomance de se utilizar uma hospedagem gratuita ao invés de um serviço de hospedagem na nuvem, não será sentido, e ainda possui a vantagem de rodar diretamente do repositório.

## Capacidade Operacional

Pela natureza da aplicação, não é esperado um grande trafégo de pessoas ao nível de congestionamento, as pessoas irão acessar inicialmente pela curiosidade de saber dos efeitos que as músicas tem nelas, porém logo essa carga será estabilizada, fazendo com que tenham poucos niveis de stress no servidor. Com a carga estabilizada, é esperado poucas falhas por parte do servidor e também que todas as requisições consigam ser processadas em menos de 1 segundo. Em uma possível situação onde o servidor esteja lidando com a mesma quantidade de entrada e saída de usuários, alguns erros podem acontecer e a espera pela resposta da requisição pode ser de 10 a 20 segundos. Em caso de uma sobrecarga, o servidor começará a devolver uma quantidade de erros consideraveis, sob um tempo instável, podendo a chegar a 1 minuto dependendo da sobrecarga.

Segue algumas prints de testes de stress realizado utilizando o Locust.



## Teste

### Avaliar impactos da música (CT01)

**Sumário**: O usuário deve ser capaz de preencher os formulários e ter um retorno em sua tela

**Executor**: Usuário

**Pré-condição**: Nenhuma

1. O usuário deve preencher o formulário
2. O usuário deve apertar o botão "Enviar"
3. O feedback baseado nas respostas do usuário deve ser exibido

**Resultado Esperado**: A resposta deve aparecer sem erros

# Apresentação da solução

Nesta seção, um vídeo de, no máximo, 5 minutos onde deverá ser descrito o escopo todo do projeto, um resumo do trabalho desenvolvido, incluindo a comprovação de que a implantação foi realizada e, as conclusões alcançadas.

